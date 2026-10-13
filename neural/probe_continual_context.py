"""Prequential native sensory prediction; answer labels never enter learning."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import psutil
from liquid_core import BLANK,load_core
from online_context import ContextLearner,RankOneContextLearner
from strength_credit import ExactStrengthLearner
from research_resources import research_slot
from train_liquid_core import write_report,prepare,evaluate

ROOT=Path(__file__).resolve().parents[1]


def trace(core, rows, device, learner=None, constant=None, shuffled=False, residual=False):
    losses=[];persistence=[];mean_errors=[];timings=[];codes=[];drifts=[];comparable=[]
    # Reset only at independent recording boundaries, never at a question.
    for row in rows:
        state=core.start().detach().clone();previous=None
        rank_credit=isinstance(learner,(RankOneContextLearner,ExactStrengthLearner))
        if rank_credit:learner.reset_credit()
        for event in row['events']:
            begin=time.perf_counter();tokens=event['tokens'].to(device).float()
            with torch.no_grad():
                target=core.sensory_target(tokens)
                # Persistence needs a previous packet; record which packets it scores.
                comparable.append(previous is not None)
                # Persistence-anchored forecast: last packet plus a state-predicted change.
                base=(previous if previous is not None else constant) if residual else None
                if previous is not None:persistence.append(float(F.mse_loss(previous,target)))
                if constant is not None:mean_errors.append(float(F.mse_loss(constant,target)))
                previous=target
            if rank_credit:
                state,code,measurement=learner.step(state,tokens,event['dt'])
                losses.append(measurement['priorPredictionMse']);drifts.append(measurement['parameterRmsDrift'])
            else:
                trace_gradients=learner is not None and learner.temporal_credit
                with torch.set_grad_enabled(trace_gradients):
                    for _ in range(10):state=core.advance(state,event['dt']/10)
                    prior=state.clone()
                    if shuffled:prior=prior.flatten(1).roll(347,1).reshape_as(prior)
                with torch.no_grad():
                    forecast=core.predict_sensory(prior)
                    if base is not None:forecast=forecast+base
                    losses.append(float(F.mse_loss(forecast,target)))
                if learner is not None:drifts.append(learner.update(prior,tokens,base)['parameterRmsDrift'])
                state=state.detach()
                with torch.set_grad_enabled(trace_gradients):
                    state=core.observe(state,tokens,event['dt'])
                    with torch.no_grad():code=int(core.logits(state).argmax(-1))
                    if code!=BLANK:state=core.feedback(state,code)
            codes.append(code)
            if device.startswith('cuda'):torch.cuda.synchronize()
            timings.append(time.perf_counter()-begin)
            if psutil.virtual_memory().used>50_000_000_000:raise MemoryError('System RAM budget exceeded.')
    def mean(values):return sum(values)/len(values) if values else None
    def matched(values):return mean([value for value,keep in zip(values,comparable) if keep])
    model_matched=matched(losses);persistence_mse=mean(persistence)
    return dict(packets=len(losses),priorPredictionMse=mean(losses),persistenceMse=persistence_mse,
        constantMse=mean(mean_errors),matchedPackets=sum(comparable),matchedPriorPredictionMse=model_matched,
        matchedConstantMse=matched(mean_errors) if mean_errors else None,
        persistenceSkillScore=1-model_matched/persistence_mse if model_matched is not None and persistence_mse else None,packetSecondsMean=mean(timings),packetSecondsMax=max(timings),
        nonblankCodes=sum(code!=BLANK for code in codes),maximumParameterRmsDrift=max(drifts,default=0.)),codes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--device',default='cpu');parser.add_argument('--passes',type=int,default=2)
    parser.add_argument('--coupling',type=float,default=.02)
    parser.add_argument('--learning-rate',type=float,default=.0001)
    parser.add_argument('--learn-writes',action='store_true')
    parser.add_argument('--temporal-credit',action='store_true',help='One-packet gradient through context only; matched control for write learning.')
    parser.add_argument('--uoro',action='store_true',help='Experimental rank-one temporal sensitivity, without input replay.')
    parser.add_argument('--exact-strength',action='store_true',help='Full strength-parameter sensitivity, with the existing predictor frozen.')
    parser.add_argument('--derivative-chunk',type=int,default=256)
    parser.add_argument('--residual',action='store_true',help='Predict the change from the last packet; starts exactly at persistence.')
    parser.add_argument('--credit-reset-each-packet',action='store_true',help='Matched short-credit control for --uoro.')
    args=parser.parse_args()
    if args.passes<1:parser.error('--passes must be positive')
    if args.learning_rate<=0:parser.error('--learning-rate must be positive')
    if not 0<=args.coupling<=1:parser.error('--coupling must be between zero and one')
    if args.credit_reset_each_packet and not (args.uoro or args.exact_strength):parser.error('--credit-reset-each-packet requires --uoro or --exact-strength')
    if args.uoro and args.exact_strength:parser.error('Select one temporal-credit implementation.')
    if args.residual and (args.uoro or args.exact_strength):parser.error('--residual supports the local and one-packet learners.')
    if not 1<=args.derivative_chunk<=256:parser.error('--derivative-chunk must be between 1 and 256')
    with research_slot('shared-context continual prediction',reserve_bytes=1_000_000_000):
        torch.set_num_threads(2);torch.manual_seed(2237)
        folder=ROOT/'artifacts'/f'liquid-continual-context-{time.time_ns()}';folder.mkdir()
        receipt=folder/'experiment.json';report=dict(state='preparing',publishedToLive=False,
            sourceCheckpoint=args.checkpoint,protocol='prequential-before-write-native-target-v1'+('-residual' if args.residual else ''),
            forecast='last packet + state-predicted change' if args.residual else 'state-predicted packet',
            coupling=args.coupling,learningRate=args.learning_rate,
            learnsWriteStrength=args.learn_writes or args.exact_strength,
            learnsContext=not args.exact_strength,
            trustRegion='Per-trained-tensor RMS drift <= 0.01 from run anchor',
            temporalCreditPackets=None if (args.uoro or args.exact_strength) and not args.credit_reset_each_packet else (1 if args.learn_writes or args.temporal_credit or args.uoro or args.exact_strength else 0),
            temporalCreditAlgorithm='structured-strength-rtrl' if args.exact_strength else ('uoro-rank1' if args.uoro else 'local-or-one-packet'),
            creditResetEachPacket=args.credit_reset_each_packet,
            updateOrder='after-assimilation' if args.uoro or args.exact_strength else 'before-assimilation',
            limitations=['Pooled projected packets can hide modality-specific errors.',
                'Recording identities/scenes overlap the historical supervised corpus.',
                'One own code per packet is a diagnostic schedule, not the live 30 ms output schedule.',
                'Prediction and adaptation alone do not establish self-awareness or broad language.'])
        write_report(receipt,report)
        try:
            corpus=json.loads((ROOT/'artifacts/workspace-raw-corpus.json').read_text())
            rows=[(item,torch.load(ROOT/item['file'],map_location='cpu',weights_only=False)) for item in corpus['cases']]
            train=[row for item,row in rows if item['split']=='train'];held=[row for item,row in rows if item['split']!='train']
            core,_=load_core(args.checkpoint,args.device);core.eval().requires_grad_(False)
            with torch.no_grad():
                constant=torch.cat([core.sensory_target(e['tokens'].to(args.device).float())
                    for row in train for e in row['events']]).mean(0,keepdim=True)
            report['frozen'],before=trace(core,held,args.device,constant=constant,residual=args.residual)
            held_items=[item for item,row in rows if item['split']!='train']
            retained,idle=prepare(core,held_items,args.device)
            report['answerRetentionBefore']=evaluate(core,retained,idle,args.device)
            def make_learner():
                if args.exact_strength:
                    return ExactStrengthLearner(core,learning_rate=args.learning_rate,chunk_size=args.derivative_chunk,reset_each_packet=args.credit_reset_each_packet)
                kind=RankOneContextLearner if args.uoro else ContextLearner
                extra={'reset_each_packet':args.credit_reset_each_packet} if args.uoro else {}
                return kind(core,learning_rate=args.learning_rate,coupling=args.coupling,train_writes=args.learn_writes,temporal_credit=args.temporal_credit,**extra)
            learner=make_learner()
            report['effectiveContextGains']=core.context_gain.detach().cpu().tolist()
            report['learnedParameterNames']=[name for name,p in core.named_parameters() if p.requires_grad]
            report['training']=[]
            for iteration in range(args.passes):
                result,_=trace(core,train,args.device,learner,constant,residual=args.residual)
                report['training'].append(result);report.update(state='learning',completedPasses=iteration+1)
                write_report(receipt,report);print(json.dumps(result),flush=True)
            core.requires_grad_(False)
            report['heldFrozenAfterLearning'],after=trace(core,held,args.device,constant=constant,residual=args.residual)
            report['statePermutationControl'],_=trace(core,held,args.device,constant=constant,shuffled=True,residual=args.residual)
            report['changedHeldCodes']=sum(a!=b for a,b in zip(before,after))
            retained,idle=prepare(core,held_items,args.device)
            report['answerRetentionAfter']=evaluate(core,retained,idle,args.device)
            snapshot=copy.deepcopy(core.state_dict())
            learner=make_learner()
            report['heldOnlineAdaptation'],_=trace(core,held,args.device,learner,constant,residual=args.residual)
            # Store pre-held-out weights, avoiding accidental held-set publication.
            torch.save(dict(architecture=core.architecture,state_dict=snapshot,
                trainingScope='Experimental label-free shared-context prediction',update=args.passes),folder/'candidate.pt')
            report.update(state='complete',trainRecordings=len(train),heldRecordings=len(held))
            write_report(receipt,report);print(str(receipt),flush=True)
        except Exception as error:
            report.update(state='failed',error=str(error));write_report(receipt,report);raise


if __name__=='__main__':main()
