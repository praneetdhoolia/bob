"""Real local HTTP requests against a synthetic runtime; no live-state mutation."""
import base64
import importlib.util
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[1]


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('bob_test_server',ROOT/'scripts/serve.py')
        cls.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.module)
        class Runtime:
            running=False
            def __init__(self):self.errors=[];self.messages=[]
            def snapshot(self):return dict(lastEvent=10)
            def event(self,kind,**data):self.errors.append((kind,data))
            def message(self,text):
                if not isinstance(text,str):raise ValueError('Text must be a string.')
                self.messages.append(text)
        cls.runtime=Runtime();cls.module.runtime=cls.runtime
        cls.server=cls.module.Server(('127.0.0.1',0),cls.module.Handler)
        cls.module.PORT=cls.server.server_address[1]
        cls.thread=threading.Thread(target=cls.server.serve_forever);cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join(timeout=2)

    def send(self,payload,token=True):
        headers={'Content-Type':'application/json'}
        if token:headers['X-Bob-Token']=self.module.TOKEN
        request=Request(f'http://127.0.0.1:{self.module.PORT}/api/message',
            data=json.dumps(payload).encode(),headers=headers,method='POST')
        try:
            with urlopen(request,timeout=3) as response:return response.status,json.load(response)
        except HTTPError as error:return error.code,json.load(error)

    def test_non_object_payloads_return_json_errors(self):
        for payload in [[],None,'text',12]:
            status,body=self.send(payload)
            self.assertEqual(status,400);self.assertIn('JSON object',body['error'])

    def test_token_and_valid_message_contract(self):
        self.assertEqual(self.send({'text':'Test only'},token=False)[0],403)
        self.assertEqual(self.send({'text':'Test only'})[0],202)
        self.assertEqual(self.runtime.messages[-1],'Test only')

    def test_sse_reconnect_cursor_rejects_malformed_header(self):
        request=Request(f'http://127.0.0.1:{self.module.PORT}/api/events?after=0',
                        headers={'Last-Event-ID':'invalid'})
        with self.assertRaises(HTTPError) as caught:urlopen(request,timeout=3)
        self.assertEqual(caught.exception.code,400)
        self.assertIn('cursor',json.load(caught.exception)['error'])

    def test_cross_site_requests_are_refused_and_responses_not_embeddable(self):
        request=Request(f'http://127.0.0.1:{self.module.PORT}/api/health',headers={'Sec-Fetch-Site':'cross-site'})
        with self.assertRaises(HTTPError) as caught:urlopen(request,timeout=3)
        self.assertEqual(caught.exception.code,403)
        with urlopen(f'http://127.0.0.1:{self.module.PORT}/favicon.ico',timeout=3) as response:
            self.assertEqual(response.headers['Cross-Origin-Resource-Policy'],'same-origin')

    def test_decompression_bomb_frame_returns_json_error(self):
        from PIL import Image
        import io
        import zlib
        stream=io.BytesIO();Image.new('L',(1,1)).save(stream,format='PNG')
        raw=bytearray(stream.getvalue())
        # Declare a 20000 x 20000 image in the header; PIL refuses it before decoding.
        raw[16:24]=(20000).to_bytes(4,'big')+(20000).to_bytes(4,'big')
        raw[29:33]=zlib.crc32(bytes(raw[12:29])).to_bytes(4,'big')
        request=Request(f'http://127.0.0.1:{self.module.PORT}/api/frame',method='POST',
            data=json.dumps({'image':base64.b64encode(bytes(raw)).decode()}).encode(),
            headers={'Content-Type':'application/json','X-Bob-Token':self.module.TOKEN})
        with self.assertRaises(HTTPError) as caught:urlopen(request,timeout=3)
        self.assertEqual(caught.exception.code,400)
        self.assertIn('pixel limit',json.load(caught.exception)['error'])

    def test_bad_training_manifest_returns_json_service_error(self):
        with patch.object(Path,'read_text',return_value='invalid'):
            with self.assertRaises(HTTPError) as caught:
                urlopen(f'http://127.0.0.1:{self.module.PORT}/api/training',timeout=3)
            self.assertEqual(caught.exception.code,503)
            self.assertIn('error',json.load(caught.exception))


if __name__=='__main__':unittest.main()
