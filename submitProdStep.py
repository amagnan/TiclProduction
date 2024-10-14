#!/usr/bin/env python3

#runTheMatrix step
#runTheMatrix.py -w upgrade -l 24805.0 -j 0
#!!!! Edit step2 and step3 config to run all = "-1" events !!!!!!!

#Front of HGCAL
#for eta in 17 19 21 23 25 27; do
#for pT in 5 10 20 50 100 150; do python3 submitProdStep.py --nRuns=10 -n 1000 -d SinglePhotons -w 24805.0_SingleGammaPt35+2026D98/ -o ProdTiclv4 -e /eos/cms/store/user/amagnan/HGCAL/TiCLv4 --pT=${pT} --Eta=${eta} --PtEta=pt${pT}_eta${eta};  done
#for pT in 5 10 20 50 100 150; do python3 submitProdStep.py -S --nRuns=10 -n 1000 -d SinglePhotons -w 24805.0_SingleGammaPt35+2026D98/ -o ProdTiclv4 -e /eos/cms/store/user/amagnan/HGCAL/TiCLv4 --pT=${pT} --PtEta=pt${pT})randEta;  done

#--config3=../step3_EMiterOnly_noPU.py

#step TiCL only -- new sub example
#pions
#for eta in 21; do for pT in 10 20 50 100 150 200; do python3 submitProdStep.py --nRuns=10 -n 0 -d ChargedPionsFromVtx --skip-step1  --skip-step2  --skip-step3 --configTicl=step3ticl_noPU.py  -o ProdTiclAll/ -e /eos/cms/store/user/amagnan/HGCAL/TiCL/ -E /eos/cms/store/user/amagnan/HGCAL/TiCL/ChargedPionsFromVtx --PtEta=pt${pT}_eta${eta} ; done; done

import subprocess
import os,sys
import optparse
import argparse
import math
import random
import numpy as np

def gen_uniform_int_random_seeds_(low, high, size):
    np.random.seed()
    r = np.random.uniform(low=low, high=high, size=size)
    return [int(x) for x in r]


usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option(      '--nRuns'       ,    dest='nRuns'              , help='number of run, 0-indexed'     , default=-1 ,      type=int)
parser.add_option('-n', '--nevts'       ,    dest='nevts'              , help='number of events to generate' , default=100,      type=int)
parser.add_option('-s', '--randomSeed'  ,    dest='randomSeed'         , help='random seed for generator process' , default=0,    type=int)
parser.add_option('-d', '--datatype'    ,    dest='datatype'           , help='data type or particle to shoot', default='SinglePhotons')
parser.add_option('-w', '--workflow'    ,    dest='workflow'           , help='Directory of workflow from runTheMatrix', default='24805.0_SingleGammaPt35+2026D98')
parser.add_option('--config1'      ,  dest='config1'           , help='Step1 config file to run', default='../CloseByParticle_Photon_ERZRanges_cfi_GEN_SIM.py')
parser.add_option('--config2'      ,  dest='config2'           , help='Step2 config file to run', default='step2_DIGI_L1TrackTrigger_L1_DIGI2RAW_HLT.py')
parser.add_option('--config3'      ,  dest='config3'           , help='Step3 config file to run', default='step3_RAW2DIGI_RECO_RECOSIM_PAT_VALIDATION_DQM.py')
parser.add_option('--configTicl'      ,  dest='configTicl'           , help='Step3 config file to rerun just TICL', default='step3ticl_noPU.py')
parser.add_option('-o', '--out'         ,    dest='out'                , help='output directory'             , default=os.getcwd() )
parser.add_option('-e', '--eosout'         ,    dest='eosout'                , help='eos path to save root file to EOS',         default='')
parser.add_option('-E', '--eosin'         ,    dest='eosin'                , help='eos path to read input root file from EOS',         default='')
parser.add_option('-S', '--no-submit'   ,    action="store_true",  dest='nosubmit'           , help='Do not submit batch job.')
parser.add_option('--skip-step1'   ,    action="store_true",  dest='skipStep1'           , help='Skip first step: copy step1 file from eos.')
parser.add_option('--skip-step2'   ,    action="store_true",  dest='skipStep2'           , help='Skip second step: copy step2 file from eos.')
parser.add_option('--skip-step3'   ,    action="store_true",  dest='skipStep3'           , help='Skip third step: copy step3 file from eos and rerun TICL.')
parser.add_option('-G', '--gridProxy',  action="store_true",  dest='gridProxy'           , help='initialise grid proxy')

parser.add_option('--pT'  ,    dest='pT'         , help='pT for generator process' , default=-1,    type=float)
parser.add_option('--minZ'  ,    dest='minZ'         , help='Minimum Zpos for generator process' , default=320,    type=float)
parser.add_option('--minPt'  ,    dest='minPt'         , help='Minimum En for generator process' , default=4.95,    type=float)
parser.add_option('--maxPt'  ,    dest='maxPt'         , help='Maximum En for generator process' , default=5.05,    type=float)
parser.add_option('--minEn'  ,    dest='minEn'         , help='Minimum En for generator process' , default=4.95,    type=float)
parser.add_option('--maxEn'  ,    dest='maxEn'         , help='Maximum En for generator process' , default=5.05,    type=float)
parser.add_option('--Eta'  ,    dest='Eta'         , help='Eta*10 for generator process' , default=17,    type=float)
parser.add_option('--Phi'  ,    dest='Phi'         , help='Phi in degrees for generator process' , default=0,    type=float)
parser.add_option('--minEta'  ,    dest='minEta'         , help='Minimum Eta for generator process' , default=1.6,    type=float)
parser.add_option('--maxEta'  ,    dest='maxEta'         , help='Maximum Eta for generator process' , default=2.8,    type=float)
parser.add_option('--minPhi'  ,    dest='minPhi'         , help='Minimum Phi for generator process' , default=-3.14159265359,    type=float)
parser.add_option('--maxPhi'  ,    dest='maxPhi'         , help='Maximum Phi for generator process' , default=3.14159265359,    type=float)
parser.add_option('--pdgid'  ,    dest='pdgid'         , help='PDG ID generator process' , default=22,    type=int)
parser.add_option('--PtEta'         ,    dest='PtEta'                , help='Pt-eta string path to save root file to EOS',         default='pt5_eta17')
parser.add_option('--EnEta'         ,    dest='EnEta'                , help='En-eta string path to save root file to EOS',         default='En5_eta17')

(opt, args) = parser.parse_args()

isPU=False
if 'WithPU' in opt.datatype:
    isPU=True

if (opt.pT>0):
    opt.minPt=0.999*opt.pT
    opt.maxPt=1.001*opt.pT
    
outDirSub='%s/%s/%s/%s'%(os.getcwd(),opt.out,opt.datatype,opt.EnEta)
eosDir='%s/%s'%(opt.eosout,opt.datatype)
eosDirIn='%s'%(opt.eosin)

eoscp='eos cp'
eosls='eos ls'
eosmk='eos mkdir'
if '/eos/user' in opt.eosin: 
    eoscp='cp'
if '/eos/user' in opt.eosout:
    eosls='ls'
    eosmk='mkdir'


gen_kwargs = dict(low=0, high=100000, size=opt.nRuns)
seeds = gen_uniform_int_random_seeds_(**gen_kwargs)

if (opt.nRuns>=1) : 
    for run in range(opt.nRuns):
        outDir='%s/run_%d'%(outDirSub,run)
        os.system('mkdir -p %s'%outDir)
else: 
    outDir=outDirSub
    os.system('mkdir -p %s'%outDir)

if (opt.randomSeed > 0):
    seeds[0] = opt.randomSeed

if not opt.skipStep1 :
    print(' -- Random seeds set to ',seeds)

#To access PU files on the grid
#os.environ['X509_USER_PROXY'] = "%s/.gridproxy.pem"%(os.environ['HOME'])
#export X509_USER_PROXY=/afs/cern.ch/user/${USER:0:1}/${USER}/x509up_u${UID} # if you plan to run skims with HTCondor

if (opt.gridProxy) :
    os.system('voms-proxy-init --voms cms --valid 168:00')

os.system('voms-proxy-info')

#wrapper
labels=('run','seed')
#It has to be $(Step) for condor submit to understand ! 
#Could be also $Process if want unique identifier when submitting parallel jobs. 
#Step will go from 0 to nJobs-1 when doing > queue nJobs.
tags=('Step','SEED')
scriptFile = open('%s/runJob.sh'%(outDirSub), 'w')
scriptFile.write('#!/bin/bash\n')
scriptFile.write('echo "- STARTING of runJob: " >> runJob.log\n')
scriptFile.write('ARGS=`getopt -o "" -l ",run:,seed:" -n "getopts_${0}" -- "$@"`\n')
scriptFile.write('echo "-- Parsing arguments : " ${ARGS} >> runJob.log\n')
scriptFile.write('eval set -- "$ARGS"\n')
scriptFile.write('while true; do\n')
scriptFile.write('case "$1" in\n')
for l,t in zip(labels, tags):
    scriptFile.write('--'+l+')\n')
    scriptFile.write('if [ -n "$2" ]; then\n')
    scriptFile.write('{}="${{2}}";\n'.format(t))
    scriptFile.write('echo "'+l+': ${'+t+'}" >> runJob.log;\n')
    scriptFile.write('fi\n')
    scriptFile.write('shift 2;;\n')
scriptFile.write('--)\n')
scriptFile.write('shift\n')
scriptFile.write('break;;\n')
scriptFile.write('esac\n')
scriptFile.write('done\n\n')

scriptFile.write('localdir=`pwd`\n')
scriptFile.write('export HOME=%s\n'%(os.environ['HOME']))
scriptFile.write('export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch \n')
scriptFile.write('source $VO_CMS_SW_DIR/cmsset_default.sh \n')
scriptFile.write('cd %s/../\n'%(os.getcwd()))
scriptFile.write('eval `scramv1 runtime -sh`\n')
scriptFile.write('cd $localdir\n')

outTag='_%s'%(opt.EnEta)
if (opt.nRuns>=1) : 
    outTag='%s_run${Step}'%(outTag)
    
if not opt.skipStep1:
    scriptFile.write('echo "-- Random seed is set to : " ${SEED} >> runJob.log\n')
    scriptFile.write('echo "cmsRun %s/%s/%s maxEvents=%d generatorRandomSeed=${SEED} minEn=%3.3f maxEn=%3.3f minEta=%3.3f maxEta=%3.3f minPhi=%3.3f maxPhi=%3.3f pdgid=%d minz=%3.3f" >> runJob.log\n'%(os.getcwd(),opt.workflow,opt.config1,opt.nevts,opt.minEn,opt.maxEn,opt.minEta,opt.maxEta,opt.minPhi,opt.maxPhi,opt.pdgid,opt.minZ))
    scriptFile.write('cmsRun %s/%s/%s maxEvents=%d generatorRandomSeed=${SEED} minEn=%3.3f maxEn=%3.3f minEta=%3.3f maxEta=%3.3f minPhi=%3.3f maxPhi=%3.3f pdgid=%d minz=%3.3f\n'%(os.getcwd(),opt.workflow,opt.config1,opt.nevts,opt.minEn,opt.maxEn,opt.minEta,opt.maxEta,opt.minPhi,opt.maxPhi,opt.pdgid,opt.minZ))

elif not opt.skipStep2:
    scriptFile.write('%s %s/step1%s.root step1.root\n'%(eoscp,eosDirIn,outTag))

if not opt.skipStep2 :
    scriptFile.write('cmsRun %s/%s/%s\n'%(os.getcwd(),opt.workflow,opt.config2))
elif not opt.skipStep3 :
    scriptFile.write('%s %s/step2%s.root step2.root\n'%(eoscp,eosDirIn,outTag))

if not opt.skipStep3 :
    scriptFile.write('cmsRun %s/%s/%s\n'%(os.getcwd(),opt.workflow,opt.config3))
else:
    scriptFile.write('%s %s/step3%s.root step3.root\n'%(eoscp,eosDirIn,outTag))
    scriptFile.write('cmsRun %s/%s\n'%(os.getcwd(),opt.configTicl))


scriptFile.write('echo "--Local directory is " $localdir >> runJob.log\n')
scriptFile.write('voms-proxy-info >> runJob.log\n')
scriptFile.write('echo home=$HOME >> runJob.log\n')
scriptFile.write('echo path=$PATH >> runJob.log\n')
scriptFile.write('echo ldlibpath=$LD_LIBRARY_PATH >> runJob.log\n')
scriptFile.write('ls -ltrh * >> runJob.log\n')
if len(opt.eosout)>0:
    scriptFile.write('%s -p %s\n'%(eosmk,eosDir))
    if not opt.skipStep1 and not opt.skipStep2 and not opt.skipStep3:
        scriptFile.write('rm step3_inDQM.root\n')
        scriptFile.write('rm step3_inMINIAODSIM.root\n')
        scriptFile.write('for idx in 1 2 3; do\n') 
    elif not opt.skipStep2 and not opt.skipStep3:
        scriptFile.write('rm step1.root\n') 
        scriptFile.write('rm step3_inDQM.root\n')
        scriptFile.write('rm step3_inMINIAODSIM.root\n')
        scriptFile.write('for idx in 2 3; do\n') 
    elif not opt.skipStep3 :
        scriptFile.write('rm step2.root\n') 
        scriptFile.write('rm step3_inDQM.root\n')
        scriptFile.write('rm step3_inMINIAODSIM.root\n')
        scriptFile.write('for idx in 3; do\n') 
    else :
        scriptFile.write('rm step3.root\n') 
        scriptFile.write('for idx in 3ticl; do\n') 

    #scriptFile.write('for outfile in step${idx} step${idx}_inDQM step${idx}_inMINIAODSIM; do\n')
    scriptFile.write('for outfile in step${idx}; do\n')
    scriptFile.write('%s ${outfile}.root %s/${outfile}%s.root\n'%(eoscp,eosDir,outTag))
    scriptFile.write('if (( "$?" != "0" )); then\n')
    scriptFile.write('echo " --- Problem with copy of file ${outfile}.root to EOS. Keeping locally." >> runJob.log\n')
    scriptFile.write('else\n')
    scriptFile.write('eossize=`%s -l %s/${outfile}%s.root | awk \'{print $5}\'`\n'%(eosls,eosDir,outTag))
    scriptFile.write('localsize=`ls -l ${outfile}.root | awk \'{print $5}\'`\n')
    scriptFile.write('if [ $eossize != $localsize ]; then\n')
    scriptFile.write('echo " --- Copy of ${outfile} file to eos failed. Localsize = $localsize, eossize = $eossize. Keeping locally..." >> runJob.log\n')
    scriptFile.write('else\n')
    scriptFile.write('echo " --- Size check done: Localsize = $localsize, eossize = $eossize" >> runJob.log\n')
    scriptFile.write('echo " --- File ${outfile}.root successfully copied to EOS: %s/${outfile}%s.root" >> runJob.log\n'%(eosDir,outTag))
    scriptFile.write('rm ${outfile}.root\n')
    scriptFile.write('fi\n')
    scriptFile.write('fi\n')
    scriptFile.write('done\n')
    scriptFile.write('done\n')

if (opt.nRuns>=1) : 
    scriptFile.write('cp * %s/run_${Step}/.\n'%(outDirSub))
else:
    scriptFile.write('cp * %s/.\n'%(outDirSub))
scriptFile.write('echo "All done"\n')
scriptFile.close()

print('Getting proxy')
subprocess.check_output(['voms-proxy-info','-path'])

proxyPath=os.popen('voms-proxy-info -path')
proxyPath=proxyPath.readline().strip()

#submit
condorFile = open('%s/condorSubmitProd.sub'%(outDirSub), 'w')
condorFile.write('x509userproxy = $ENV(X509_USER_PROXY)\n')
condorFile.write('use_x509userproxy = True\n')
condorFile.write('universe = vanilla\n')
if (not opt.skipStep1) or isPU:
    condorFile.write('+JobFlavour = "nextweek"\n')
else:
    condorFile.write('+JobFlavour = "tomorrow"\n')
condorFile.write('Executable = %s/runJob.sh\n'%outDirSub)
if not opt.skipStep1 :
    condorFile.write('Arguments = --run $(Process) --seed $(SEED)\n')
    if (opt.nRuns>=1) : 
        condorFile.write('Output = %s/run_$(Process)/condor.out\n'%outDirSub)
        condorFile.write('Error = %s/run_$(Process)/condor.err\n'%outDirSub)
        condorFile.write('Log = %s/run_$(Process)/condor.log\n'%outDirSub)
    else:
        condorFile.write('Output = %s/condor.out\n'%outDirSub)
        condorFile.write('Error = %s/condor.err\n'%outDirSub)
        condorFile.write('Log = %s/condor.log\n'%outDirSub)
elif (opt.nRuns>=1):
    condorFile.write('Arguments = --run $(Step)\n')
    condorFile.write('Output = %s/run_$(Step)/condor.out\n'%outDirSub)
    condorFile.write('Error = %s/run_$(Step)/condor.err\n'%outDirSub)
    condorFile.write('Log = %s/run_$(Step)/condor.log\n'%outDirSub)
else:
    condorFile.write('Output = %s/condor.out\n'%outDirSub)
    condorFile.write('Error = %s/condor.err\n'%outDirSub)
    condorFile.write('Log = %s/condor.log\n'%outDirSub)

if not opt.skipStep1 :
    condorFile.write('Queue 1 SEED from (\n')
    for run in range(opt.nRuns):
        seed=seeds[run]
        condorFile.write('{}\n'.format(seed))
    condorFile.write(')')
else:
    condorFile.write('Queue %d\n'%opt.nRuns)

condorFile.close()

os.system('chmod u+rwx %s/runJob.sh'%outDirSub)
if opt.nosubmit : os.system('echo condor_submit %s/condorSubmitProd.sub'%(outDirSub)) 
else: 
    os.system('echo submitting job %s'%(outDirSub))
    os.system('condor_submit %s/condorSubmitProd.sub'%(outDirSub))

