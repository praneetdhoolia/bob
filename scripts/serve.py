"""One local process and one port for the liquid agent and its interface."""
import base64
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import queue
import secrets
import socket
import sys
import threading
import time
from urllib.parse import urlsplit,parse_qs

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];PORT=3217;TOKEN=secrets.token_urlsafe(32)
# Each event stream holds one server thread; bound them.
EVENT_STREAMS=threading.BoundedSemaphore(8)
sys.path.insert(0,str(ROOT/'neural'))
from liquid_runtime import LiquidRuntime


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass

    def reply(self,status,value,kind='application/json'):
        payload=json.dumps(value).encode() if kind=='application/json' else value
        self.send_response(status);self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        # Other sites must not embed camera frames or responses, even without CORS.
        self.send_header('Cross-Origin-Resource-Policy','same-origin')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers();self.wfile.write(payload)

    def trusted(self):
        if self.headers.get('Sec-Fetch-Site') in ['cross-site','same-site']:return False
        return self.headers.get('Host') in [f'127.0.0.1:{PORT}',f'localhost:{PORT}'] and self.headers.get('Origin') in [None,f'http://127.0.0.1:{PORT}',f'http://localhost:{PORT}']

    def do_GET(self):
        if not self.trusted():return self.reply(403,{'error':'Unexpected origin.'})
        parsed=urlsplit(self.path);path=parsed.path
        try:
            if path=='/api/session':return self.reply(200,{'token':TOKEN})
            if path=='/api/status':return self.reply(200,runtime.snapshot())
            if path=='/api/state':return self.reply(200,runtime.snapshot(full=True,include_messages=False))
            if path=='/api/health':return self.reply(200,{'pid':os.getpid(),'ready':runtime.ready})
            if path=='/api/training':
                return self.reply(200,json.loads((ROOT/'artifacts/liquid-core-training.json').read_text(encoding='utf8')))
            if path=='/api/frame':
                with runtime.lock:data=runtime.latest_jpeg
                return self.reply(200,data,'image/jpeg') if data else self.reply(404,{'error':'No processed frame yet.'})
            if path=='/api/events':
                try:after=max(0,int(self.headers.get('Last-Event-ID') or parse_qs(parsed.query).get('after',['0'])[0]))
                except ValueError:return self.reply(400,{'error':'Invalid event cursor.'})
                if not EVENT_STREAMS.acquire(blocking=False):return self.reply(429,{'error':'Too many event streams.'})
                try:self.stream_events(after)
                finally:EVENT_STREAMS.release()
                return
            files={'/':('agent.html','text/html; charset=utf-8'),'/agent.js':('agent.js','text/javascript'),
                   '/agent.css':('agent.css','text/css'),'/audio-worklet.js':('audio-worklet.js','text/javascript')}
            if path in files:
                name,kind=files[path];return self.reply(200,(ROOT/'public'/name).read_bytes(),kind)
            if path=='/favicon.ico':return self.reply(204,b'','image/x-icon')
            return self.reply(404,{'error':'Unknown route.'})
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
        except (OSError,ValueError,KeyError,RuntimeError) as error:
            self.reply(503,{'error':str(error)})

    def stream_events(self, after):
        self.send_response(200);self.send_header('Content-Type','text/event-stream')
        self.send_header('Cache-Control','no-store');self.send_header('Cross-Origin-Resource-Policy','same-origin')
        self.end_headers()
        snapshot=runtime.snapshot();after=min(after,snapshot['lastEvent'])
        self.wfile.write(('data: '+json.dumps({'type':'snapshot','data':snapshot})+'\n\n').encode());self.wfile.flush()
        while runtime.running:
            with runtime.lock:events=[dict(event) for event in runtime.events if event['id']>after]
            for event in events:
                self.wfile.write(('id: '+str(event['id'])+'\ndata: '+json.dumps(event)+'\n\n').encode());after=event['id']
            self.wfile.write(b': heartbeat\n\n');self.wfile.flush();time.sleep(.1)

    def do_POST(self):
        if not self.trusted() or self.headers.get('X-Bob-Token')!=TOKEN:return self.reply(403,{'error':'Invalid session.'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=4_000_000:raise ValueError('Invalid request size.')
            data=json.loads(self.rfile.read(length));path=urlsplit(self.path).path
            if not isinstance(data,dict):raise ValueError('Expected a JSON object.')
            if path=='/api/message':runtime.message(data.get('text'))
            elif path=='/api/frame':
                raw=base64.b64decode(data['image'],validate=True)
                try:
                    with Image.open(io.BytesIO(raw),formats=['JPEG','PNG']) as image:runtime.receive_frame(image)
                except Image.DecompressionBombError as error:raise ValueError('Frame exceeds the pixel limit.') from error
                except SyntaxError as error:raise ValueError('Malformed image.') from error
            elif path=='/api/audio':
                if data.get('sampleRate')!=16000:raise ValueError('Expected 16 kHz float32 audio.')
                raw=base64.b64decode(data['pcm'],validate=True)
                if len(raw)%4:raise ValueError('Malformed waveform.')
                runtime.receive_audio(np.frombuffer(raw,dtype='<f4'))
            elif path=='/api/source':runtime.set_source(data.get('source'))
            elif path=='/api/active':runtime.set_active(data.get('active'))
            elif path=='/api/save':runtime.save_state()
            elif path=='/api/shutdown':
                try:
                    runtime.close();self.reply(200,{'saved':True,'stopping':True})
                except (OSError,RuntimeError,ValueError) as error:
                    self.reply(500,{'error':f'Stopping without a final save: {error}','stopping':True})
                finally:
                    # Never keep serving after the runtime has been told to stop.
                    threading.Thread(target=server.shutdown,daemon=True).start()
                return
            else:return self.reply(404,{'error':'Unknown route.'})
            return self.reply(202,{'accepted':True})
        except queue.Full as error:self.reply(429,{'error':str(error) or 'Input queue full.'})
        except (ValueError,KeyError,TypeError,OSError) as error:
            runtime.event('input-error',path=urlsplit(self.path).path,error=str(error))
            self.reply(400,{'error':str(error)})


class Server(ThreadingHTTPServer):
    daemon_threads=True;allow_reuse_address=False
    def server_bind(self):
        if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()


if __name__=='__main__':
    server=Server(('127.0.0.1',PORT),Handler)
    runtime=LiquidRuntime(ROOT)
    (ROOT/'.runtime/agent-server.json').write_text(json.dumps({'pid':os.getpid(),'port':PORT,'startedAtUnix':time.time(),'script':str(Path(__file__).resolve())}),encoding='utf8')
    print('Bob liquid agent: http://127.0.0.1:3217/',flush=True)
    try:server.serve_forever()
    finally:runtime.close();server.server_close()
