/*
 * xpmclient.cpp
 *
 *  Created on: 01.05.2014
 *      Author: mad
 */



#include "xpmclient.h"
#include "prime.h"
#include "benchmarks.h"

extern "C" {
	#include "adl.h"
}

#include "loguru.hpp"

#include <fstream>
#include <set>
#include <memory>
#include <chrono>
#if defined(__GXX_EXPERIMENTAL_CXX0X__) && (__cplusplus < 201103L)
#define steady_clock monotonic_clock
#endif  

#include <math.h>

std::vector<unsigned> gPrimes2;

double GetPrimeDifficulty(unsigned int nBits)
{
    return ((double) nBits / (double) (1 << nFractionalBits));
}

BaseClient *createClient(void *ctx)
{
  return new XPMClient(ctx);
}

PrimeMiner::PrimeMiner(unsigned id, unsigned threads, unsigned sievePerRound, unsigned depth, unsigned LSize) {
	
	mID = id;
	mThreads = threads;

  mSievePerRound = sievePerRound;
	mDepth = depth;
  mLSize = LSize;  
	
	mBlockSize = 0;
	mConfig = {0};

  _context = 0;
 	mHMFermatStream = 0;
 	mSieveStream = 0;
	mHashMod = 0;
	mSieveSetup = 0;
	mSieve = 0;
	mSieveSearch = 0;
	mFermatSetup = 0;
	mFermatKernel352 = 0;
  mFermatKernel320 = 0;  
	mFermatCheck = 0;  
	
	MakeExit = false;
	
}

PrimeMiner::~PrimeMiner() {
  if (mSieveStream)
    cuStreamDestroy(mSieveStream);
  if (mHMFermatStream)
    cuStreamDestroy(mHMFermatStream);
}

bool PrimeMiner::Initialize(CUcontext context, CUdevice device, CUmodule module)
{
  _context = context;
  cuCtxSetCurrent(context);
  
  // Lookup kernels by mangled name
  CUDA_SAFE_CALL(cuModuleGetFunction(&mHashMod, module, "_Z18bhashmodUsePrecalcjPjS_S_S_jjjjjjjjjjjj"));
  CUDA_SAFE_CALL(cuModuleGetFunction(&mSieveSetup, module, "_Z11setup_sievePjS_PKjS_jS_"));
  CUDA_SAFE_CALL(cuModuleGetFunction(&mSieve, module, "_Z5sievePjS_P5uint2"));
  CUDA_SAFE_CALL(cuModuleGetFunction(&mSieveSearch, module, "_Z7s_sievePKjS0_P8fermat_tS2_Pjjjj"));
  CUDA_SAFE_CALL(cuModuleGetFunction(&mFermatSetup, module, "_Z12setup_fermatPjPK8fermat_tS_"));
  CUDA_SAFE_CALL(cuModuleGetFunction(&mFermatKernel352, module, "_Z13fermat_kernelPhPKj"));
  CUDA_SAFE_CALL(cuModuleGetFunction(&mFermatKernel320, module, "_Z16fermat_kernel320PhPKj"));
  CUDA_SAFE_CALL(cuModuleGetFunction(&mFermatCheck, module, "_Z12check_fermatP8fermat_tPjS0_S1_PKhPKS_j"));  
  
  CUDA_SAFE_CALL(cuStreamCreate(&mSieveStream, CU_STREAM_NON_BLOCKING));
  CUDA_SAFE_CALL(cuStreamCreate(&mHMFermatStream, CU_STREAM_NON_BLOCKING));
  
  // Get miner config
  {
    CUfunction getConfigKernel;
    CUDA_SAFE_CALL(cuModuleGetFunction(&getConfigKernel, module, "_Z9getconfigP8config_t"));  
    
    cudaBuffer<config_t> config;
    CUDA_SAFE_CALL(config.init(1, false));
    void *args[] = { &config._deviceData };
    CUDA_SAFE_CALL(cuLaunchKernel(getConfigKernel,
                                  1, 1, 1,
                                  1, 1, 1,
                                  0, NULL, args, 0));
    CUDA_SAFE_CALL(cuCtxSynchronize());
    CUDA_SAFE_CALL(config.copyToHost());
    mConfig = *config._hostData;
  }

  LOG_F(INFO, "N=%d SIZE=%d STRIPES=%d WIDTH=%d PCOUNT=%d TARGET=%d",
         mConfig.N, mConfig.SIZE, mConfig.STRIPES, mConfig.WIDTH, mConfig.PCOUNT, mConfig.TARGET);
  
  int computeUnits;
  CUDA_SAFE_CALL(cuDeviceGetAttribute(&computeUnits, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  mBlockSize = computeUnits * 4 * 64;
  LOG_F(INFO, "GPU %d: has %d CUs", mID, computeUnits);
  return true;
}

void PrimeMiner::InvokeMining(void *args, void *ctx, void *pipe) {
	
	((PrimeMiner*)args)->Mining(ctx, pipe);
	
}

void PrimeMiner::FermatInit(pipeline_t &fermat, unsigned mfs)
{
  fermat.current = 0;
  fermat.bsize = 0;
  CUDA_SAFE_CALL(fermat.input.init(mfs*mConfig.N, true));
  CUDA_SAFE_CALL(fermat.output.init(mfs, true));

  for(int i = 0; i < 2; ++i){
    CUDA_SAFE_CALL(fermat.buffer[i].info.init(mfs, true));
    CUDA_SAFE_CALL(fermat.buffer[i].count.init(1, false)); // CL_MEM_ALLOC_HOST_PTR
  }
}

void PrimeMiner::FermatDispatch(pipeline_t &fermat,
                                cudaBuffer<fermat_t> sieveBuffers[SW][FERMAT_PIPELINES][2],
                                cudaBuffer<uint32_t> candidatesCountBuffers[SW][2],
                                unsigned pipelineIdx,
                                int ridx,
                                int widx,
                                uint64_t &testCount,
                                uint64_t &fermatCount,
                                CUfunction fermatKernel,
                                unsigned sievePerRound)
{
  // fermat dispatch
  {
    uint32_t &count = fermat.buffer[ridx].count[0];
    uint32_t left = fermat.buffer[widx].count[0] - fermat.bsize;
    if(left > 0){
      cuMemcpyDtoDAsync(fermat.buffer[ridx].info._deviceData + count*sizeof(fermat_t),
                        fermat.buffer[widx].info._deviceData + fermat.bsize*sizeof(fermat_t),
                        left*sizeof(fermat_t), mHMFermatStream);
      count += left;
    }
    
    for(int i = 0; i < sievePerRound; ++i){
      uint32_t &avail = (candidatesCountBuffers[i][ridx])[pipelineIdx];
      if(avail){
        cuMemcpyDtoDAsync(fermat.buffer[ridx].info._deviceData + count*sizeof(fermat_t),
                          sieveBuffers[i][pipelineIdx][ridx]._deviceData,
                          avail*sizeof(fermat_t), mHMFermatStream);
        count += avail;
        testCount += avail;
        fermatCount += avail;
        avail = 0;
      }
    }
    
    fermat.buffer[widx].count[0] = 0;
    CUDA_SAFE_CALL(fermat.buffer[widx].count.copyToDevice(mHMFermatStream));
    
    fermat.bsize = 0;
    if(count > mBlockSize){                 
      fermat.bsize = count - (count % mBlockSize);
      {
        // Fermat test setup
        void *arguments[] = {
          &fermat.input._deviceData,
          &fermat.buffer[ridx].info._deviceData,
          &hashBuf._deviceData
        };
        
        CUDA_SAFE_CALL(cuLaunchKernel(mFermatSetup,
                                      fermat.bsize/256, 1, 1,                                
                                      256, 1, 1,
                                      0, mHMFermatStream, arguments, 0));
      }
      
      {
        // Fermat test
        void *arguments[] = {
          &fermat.output._deviceData,
          &fermat.input._deviceData
        };
        
        CUDA_SAFE_CALL(cuLaunchKernel(fermatKernel,
                                      fermat.bsize/64, 1, 1,                                
                                      64, 1, 1,
                                      0, mHMFermatStream, arguments, 0));        
      }
      
      {
        // Fermat check
        void *arguments[] = {
          &fermat.buffer[widx].info._deviceData,
          &fermat.buffer[widx].count._deviceData,
          &final.info._deviceData,
          &final.count._deviceData,
          &fermat.output._deviceData,
          &fermat.buffer[ridx].info._deviceData,
          &mDepth
        };        
        
        CUDA_SAFE_CALL(cuLaunchKernel(mFermatCheck,
                                      fermat.bsize/256, 1, 1,                                
                                      256, 1, 1,
                                      0, mHMFermatStream, arguments, 0));        
      }
      
//       fermat.buffer[widx].count.copyToHost(mBig);
    } else {
      // printf(" * warning: no enough candidates available (pipeline %u)\n", pipelineIdx);
    }
    // printf("fermat: total of %d infos, bsize = %d\n", count, fermat.bsize);
  }
}

void PrimeMiner::Mining(void *ctx, void *pipe) {
  cuCtxSetCurrent(_context);
  time_t starttime = time(0);
	void* blocksub = zmq_socket(ctx, ZMQ_SUB);
	void* worksub = zmq_socket(ctx, ZMQ_SUB);
	void* statspush = zmq_socket(ctx, ZMQ_PUSH);
	void* sharepush = zmq_socket(ctx, ZMQ_PUSH);  
	
	zmq_connect(blocksub, "inproc://blocks");
	zmq_connect(worksub, "inproc://work");
	zmq_connect(statspush, "inproc://stats");
	zmq_connect(sharepush, "inproc://shares");        

	{
		const char one[2] = {1, 0};
		zmq_setsockopt (blocksub, ZMQ_SUBSCRIBE, one, 1);
		zmq_setsockopt (worksub, ZMQ_SUBSCRIBE, one, 1);
	}
	
	proto::Block block;
	proto::Work work;
	proto::Share share;
	
	block.set_height(1);
	work.set_height(0);
	
	share.set_addr(gAddr);
	share.set_name(gClientName);
	share.set_clientid(gClientID);
	
	stats_t stats;
	stats.id = mID;
	stats.errors = 0;
	stats.fps = 0;
	stats.primeprob = 0;
	stats.cpd = 0;
	
	const unsigned mPrimorial = 13;
	uint64_t fermatCount = 1;
	uint64_t primeCount = 1;
	
	time_t time1 = time(0);
	time_t time2 = time(0);
	uint64_t testCount = 0;

	unsigned iteration = 0;
	mpz_class primorial[maxHashPrimorial];
	block_t blockheader;
  search_t hashmod;
  sha256precalcData precalcData;

  lifoBuffer<hash_t> hashes(PW);
	cudaBuffer<uint32_t> sieveBuf[2];
	cudaBuffer<uint32_t> sieveOff[2];
  cudaBuffer<fermat_t> sieveBuffers[SW][FERMAT_PIPELINES][2];
  cudaBuffer<uint32_t> candidatesCountBuffers[SW][2];
  pipeline_t fermat320;
  pipeline_t fermat352;
	CPrimalityTestParams testParams;
	std::vector<fermat_t> candis;
  unsigned numHashCoeff = 32768;

  cudaBuffer<uint32_t> primeBuf[maxHashPrimorial];
  cudaBuffer<uint32_t> primeBuf2[maxHashPrimorial];
  
  CUevent sieveEvent;
  CUDA_SAFE_CALL(cuEventCreate(&sieveEvent, CU_EVENT_BLOCKING_SYNC));
  
  for (unsigned i = 0; i < maxHashPrimorial - mPrimorial; i++) {
    CUDA_SAFE_CALL(primeBuf[i].init(mConfig.PCOUNT, true));
    CUDA_SAFE_CALL(primeBuf[i].copyToDevice(&gPrimes[mPrimorial+i+1]));
    CUDA_SAFE_CALL(primeBuf2[i].init(mConfig.PCOUNT*2, true));
    CUDA_SAFE_CALL(primeBuf2[i].copyToDevice(&gPrimes2[2*(mPrimorial+i)+2]));
    mpz_class p = 1;
    for(unsigned j = 0; j <= mPrimorial+i; j++)
      p *= gPrimes[j];    
    primorial[i] = p;
  }  
  
	{
		unsigned primorialbits = mpz_sizeinbase(primorial[0].get_mpz_t(), 2);
		mpz_class sievesize = mConfig.SIZE*32*mConfig.STRIPES;
		unsigned sievebits = mpz_sizeinbase(sievesize.get_mpz_t(), 2);
    LOG_F(INFO, "GPU %d: primorial = %s (%d bits)", mID, primorial[0].get_str(10).c_str(), primorialbits);
    LOG_F(INFO, "GPU %d: sieve size = %s (%d bits)", mID, sievesize.get_str(10).c_str(), sievebits);
	}
  
  CUDA_SAFE_CALL(hashmod.midstate.init(8, false));
  CUDA_SAFE_CALL(hashmod.found.init(128, false));
  CUDA_SAFE_CALL(hashmod.primorialBitField.init(128, false));
  CUDA_SAFE_CALL(hashmod.count.init(1, false));
  CUDA_SAFE_CALL(hashBuf.init(PW*mConfig.N, false));
	
	for(int sieveIdx = 0; sieveIdx < SW; ++sieveIdx) {
    for(int instIdx = 0; instIdx < 2; ++instIdx){    
      for (int pipelineIdx = 0; pipelineIdx < FERMAT_PIPELINES; pipelineIdx++)
        CUDA_SAFE_CALL(sieveBuffers[sieveIdx][pipelineIdx][instIdx].init(MSO, true));
      
      CUDA_SAFE_CALL(candidatesCountBuffers[sieveIdx][instIdx].init(FERMAT_PIPELINES, false)); // CL_MEM_ALLOC_HOST_PTR
    }
  }
	
	for(int k = 0; k < 2; ++k){
    CUDA_SAFE_CALL(sieveBuf[k].init(mConfig.SIZE*mConfig.STRIPES/2*mConfig.WIDTH, true));
    CUDA_SAFE_CALL(sieveOff[k].init(mConfig.PCOUNT*mConfig.WIDTH, true));
	}
	
  CUDA_SAFE_CALL(final.info.init(MFS/(4*mDepth), false)); // CL_MEM_ALLOC_HOST_PTR
  CUDA_SAFE_CALL(final.count.init(1, false));	 // CL_MEM_ALLOC_HOST_PTR

  FermatInit(fermat320, MFS);
  FermatInit(fermat352, MFS);    

  cudaBuffer<uint32_t> modulosBuf[maxHashPrimorial];
  unsigned modulosBufferSize = mConfig.PCOUNT*(mConfig.N-1);   
  for (unsigned bufIdx = 0; bufIdx < maxHashPrimorial-mPrimorial; bufIdx++) {
    cudaBuffer<uint32_t> &current = modulosBuf[bufIdx];
    CUDA_SAFE_CALL(current.init(modulosBufferSize, false));
    for (unsigned i = 0; i < mConfig.PCOUNT; i++) {
      mpz_class X = 1;
      for (unsigned j = 0; j < mConfig.N-1; j++) {
        X <<= 32;
        mpz_class mod = X % gPrimes[i+mPrimorial+bufIdx+1];
        current[mConfig.PCOUNT*j+i] = mod.get_ui();
      }
    }
    
    CUDA_SAFE_CALL(current.copyToDevice());
  }    

  czmq_signal(pipe);
  czmq_poll(pipe, -1);

	bool run = true;
	while(run){
    if(czmq_poll(pipe, 0)) {
      czmq_wait(pipe);
      czmq_wait(pipe);
    }

		{
			time_t currtime = time(0);
			time_t elapsed = currtime - time1;
			if(elapsed > 11){
 				zmq_send(statspush, &stats, sizeof(stats), 0);                          
				time1 = currtime;
			}
			
			elapsed = currtime - time2;
			if(elapsed > 15){
				stats.fps = testCount / elapsed;
				time2 = currtime;
				testCount = 0;
			}
		}
		
		stats.primeprob = pow(double(primeCount)/double(fermatCount), 1./mDepth)
				- 0.0003 * (double(mConfig.TARGET-1)/2. - double(mDepth-1)/2.);
		stats.cpd = 24.*3600. * double(stats.fps) * pow(stats.primeprob, mConfig.TARGET);
		
		// get work
		bool reset = false;
		{
			bool getwork = true;
			while(getwork && run){
        if(czmq_poll(worksub, 0) || work.height() < block.height()){
					run = ReceivePub(work, worksub);
					reset = true;
				}
				
				getwork = false;
        if(czmq_poll(blocksub, 0) || work.height() > block.height()){
					run = ReceivePub(block, blocksub);
          getwork = true;
				}
			}
		}
		if(!run)
			break;
		
		// reset if new work
		if(reset){
      hashes.clear();
			hashmod.count[0] = 0;
			fermat320.bsize = 0;
			fermat320.buffer[0].count[0] = 0;
			fermat320.buffer[1].count[0] = 0;
      fermat352.bsize = 0;
      fermat352.buffer[0].count[0] = 0;
      fermat352.buffer[1].count[0] = 0;      
			final.count[0] = 0;
      
      for(int sieveIdx = 0; sieveIdx < SW; ++sieveIdx) {
        for(int instIdx = 0; instIdx < 2; ++instIdx) {
          for (int pipelineIdx = 0; pipelineIdx < FERMAT_PIPELINES; pipelineIdx++)
            (candidatesCountBuffers[sieveIdx][instIdx])[pipelineIdx] = 0;
        }
      }      
			
			blockheader.version = block_t::CURRENT_VERSION;
			blockheader.hashPrevBlock.SetHex(block.hash());
			blockheader.hashMerkleRoot.SetHex(work.merkle());
			blockheader.time = work.time() + mID;
			blockheader.bits = work.bits();
			blockheader.nonce = 1;
			testParams.nBits = blockheader.bits;
			
			unsigned target = TargetGetLength(blockheader.bits);
      precalcSHA256(&blockheader, hashmod.midstate._hostData, &precalcData);
      hashmod.count[0] = 0;
      CUDA_SAFE_CALL(hashmod.midstate.copyToDevice(mHMFermatStream));
      CUDA_SAFE_CALL(hashmod.count.copyToDevice(mHMFermatStream));
		}
		
		// hashmod fetch & dispatch
		{
// 			printf("got %d new hashes\n", hashmod.count[0]); fflush(stdout);
			for(unsigned i = 0; i < hashmod.count[0]; ++i) {
				hash_t hash;
				hash.iter = iteration;
				hash.time = blockheader.time;
				hash.nonce = hashmod.found[i];
        uint32_t primorialBitField = hashmod.primorialBitField[i];
        uint32_t primorialIdx = primorialBitField >> 16;
        uint64_t realPrimorial = 1;
        for (unsigned j = 0; j < primorialIdx+1; j++) {
          if (primorialBitField & (1 << j))
            realPrimorial *= gPrimes[j];
        }      
        
        mpz_class mpzRealPrimorial;        
        mpz_import(mpzRealPrimorial.get_mpz_t(), 2, -1, 4, 0, 0, &realPrimorial);            
        primorialIdx = std::max(mPrimorial, primorialIdx) - mPrimorial;
        mpz_class mpzHashMultiplier = primorial[primorialIdx] / mpzRealPrimorial;
        unsigned hashMultiplierSize = mpz_sizeinbase(mpzHashMultiplier.get_mpz_t(), 2);      
        mpz_import(mpzRealPrimorial.get_mpz_t(), 2, -1, 4, 0, 0, &realPrimorial);        
				
				block_t b = blockheader;
				b.nonce = hash.nonce;
				
				SHA_256 sha;
				sha.init();
				sha.update((const unsigned char*)&b, sizeof(b));
				sha.final((unsigned char*)&hash.hash);
				sha.init();
				sha.update((const unsigned char*)&hash.hash, sizeof(uint256));
				sha.final((unsigned char*)&hash.hash);
				
				if(hash.hash < (uint256(1) << 255)){
          LOG_F(WARNING, "hash does not meet minimum.\n");
					stats.errors++;
					continue;
				}
				
				mpz_class mpzHash;
				mpz_set_uint256(mpzHash.get_mpz_t(), hash.hash);
        if(!mpz_divisible_p(mpzHash.get_mpz_t(), mpzRealPrimorial.get_mpz_t())){
          LOG_F(WARNING, "mpz_divisible_ui_p failed.\n");
					stats.errors++;
					continue;
				}
				
				hash.primorialIdx = primorialIdx;
        hash.primorial = mpzHashMultiplier;
        hash.shash = mpzHash * hash.primorial;       

        unsigned hid = hashes.push(hash);
        memset(&hashBuf[hid*mConfig.N], 0, sizeof(uint32_t)*mConfig.N);
        mpz_export(&hashBuf[hid*mConfig.N], 0, -1, 4, 0, 0, hashes.get(hid).shash.get_mpz_t());        
			}
			
			if (hashmod.count[0])
        CUDA_SAFE_CALL(hashBuf.copyToDevice(mSieveStream));
			
			//printf("hashlist.size() = %d\n", (int)hashlist.size());
			hashmod.count[0] = 0;
			
      int numhash = ((int)(16*mSievePerRound) - (int)hashes.remaining()) * numHashCoeff;

			if(numhash > 0){
        numhash += mLSize - numhash % mLSize;
				if(blockheader.nonce > (1u << 31)){
					blockheader.time += mThreads;
					blockheader.nonce = 1;
          precalcSHA256(&blockheader, hashmod.midstate._hostData, &precalcData);
				}

        CUDA_SAFE_CALL(hashmod.midstate.copyToDevice(mHMFermatStream));
        CUDA_SAFE_CALL(hashmod.count.copyToDevice(mHMFermatStream));

        void *arguments[] = {
          &blockheader.nonce,
          &hashmod.found._deviceData,
          &hashmod.count._deviceData,
          &hashmod.primorialBitField._deviceData,
          &hashmod.midstate._deviceData,
          &precalcData.merkle,
          &precalcData.time,
          &precalcData.nbits,
          &precalcData.W0,
          &precalcData.W1,
          &precalcData.new1_0,
          &precalcData.new1_1,
          &precalcData.new1_2,
          &precalcData.new2_0,
          &precalcData.new2_1,
          &precalcData.new2_2,
          &precalcData.temp2_3
        };
        
        CUDA_SAFE_CALL(cuLaunchKernel(mHashMod,
                                      numhash/mLSize, 1, 1,                                
                                      mLSize, 1, 1,
                                      0, mHMFermatStream, arguments, 0));
        
				blockheader.nonce += numhash;
			}
		}

		int ridx = iteration % 2;
		int widx = ridx xor 1;
		
		// sieve dispatch    
      for (unsigned i = 0; i < mSievePerRound; i++) {
        if(hashes.empty()){
          if (!reset) {
            numHashCoeff += 32768;
            LOG_F(WARNING, "ran out of hashes, increasing sha256 work size coefficient to %u", numHashCoeff);
          }
          break;
        }
        
        int hid = hashes.pop();
        unsigned primorialIdx = hashes.get(hid).primorialIdx;    
        
        CUDA_SAFE_CALL(candidatesCountBuffers[i][widx].copyToDevice(mSieveStream));
        
        {
          void *arguments[] = {
            &sieveOff[0]._deviceData,
            &sieveOff[1]._deviceData,
            &primeBuf[primorialIdx]._deviceData,
            &hashBuf._deviceData,
            &hid,
            &modulosBuf[primorialIdx]._deviceData
          };
          
          CUDA_SAFE_CALL(cuLaunchKernel(mSieveSetup,
                                        mConfig.PCOUNT/mLSize, 1, 1,                                
                                        mLSize, 1, 1,
                                        0, mSieveStream, arguments, 0));          
				}

        {
          void *arguments[] = {
            &sieveBuf[0]._deviceData,
            &sieveOff[0]._deviceData,
            &primeBuf2[primorialIdx]._deviceData
          };
      
          CUDA_SAFE_CALL(cuLaunchKernel(mSieve,
                                        mConfig.STRIPES/2, mConfig.WIDTH, 1,                                
                                        mLSize, 1, 1,
                                        0, mSieveStream, arguments, 0));
        }
        
        {
          void *arguments[] = {
            &sieveBuf[1]._deviceData,
            &sieveOff[1]._deviceData,
            &primeBuf2[primorialIdx]._deviceData
          };
      
          CUDA_SAFE_CALL(cuLaunchKernel(mSieve,
                                        mConfig.STRIPES/2, mConfig.WIDTH, 1,                                
                                        mLSize, 1, 1,
                                        0, mSieveStream, arguments, 0));          
          
        }         
         
				{
          uint32_t multiplierSize = mpz_sizeinbase(hashes.get(hid).shash.get_mpz_t(), 2);
          void *arguments[] = {
            &sieveBuf[0]._deviceData,
            &sieveBuf[1]._deviceData,
            &sieveBuffers[i][0][widx]._deviceData,
            &sieveBuffers[i][1][widx]._deviceData,
            &candidatesCountBuffers[i][widx]._deviceData,
            &hid,
            &multiplierSize,
            &mDepth
          };
          
          CUDA_SAFE_CALL(cuLaunchKernel(mSieveSearch,
                                        (mConfig.SIZE*mConfig.STRIPES/2)/256, 1, 1,
                                        256, 1, 1,
                                        0, mSieveStream, arguments, 0));
          
          CUDA_SAFE_CALL(cuEventRecord(sieveEvent, mSieveStream)); 
				}
			}
		
    
		// get candis
		int numcandis = final.count[0];
		numcandis = std::min(numcandis, (int)final.info._size);
		numcandis = std::max(numcandis, 0);
//  		printf("got %d new candis\n", numcandis);
		candis.resize(numcandis);
		primeCount += numcandis;
		if(numcandis)
			memcpy(&candis[0], final.info._hostData, numcandis*sizeof(fermat_t));
		
    final.count[0] = 0;
    CUDA_SAFE_CALL(final.count.copyToDevice(mHMFermatStream));
    FermatDispatch(fermat320, sieveBuffers, candidatesCountBuffers, 0, ridx, widx, testCount, fermatCount, mFermatKernel320, mSievePerRound);
    FermatDispatch(fermat352, sieveBuffers, candidatesCountBuffers, 1, ridx, widx, testCount, fermatCount, mFermatKernel352, mSievePerRound);

    // copyToHost (cuMemcpyDtoHAsync) always blocking sync call!
    // syncronize our stream one time per iteration
    // sieve stream is first because it much bigger
    CUDA_SAFE_CALL(cuEventSynchronize(sieveEvent)); 
#ifdef __WINDOWS__  
    CUDA_SAFE_CALL(cuCtxSynchronize());
#endif
    for (unsigned i = 0; i < mSievePerRound; i++)
      CUDA_SAFE_CALL(candidatesCountBuffers[i][widx].copyToHost(mSieveStream));
    
    // Synchronize Fermat stream, copy all needed data
    CUDA_SAFE_CALL(hashmod.found.copyToHost(mHMFermatStream));
    CUDA_SAFE_CALL(hashmod.primorialBitField.copyToHost(mHMFermatStream));
    CUDA_SAFE_CALL(hashmod.count.copyToHost(mHMFermatStream));
    CUDA_SAFE_CALL(fermat320.buffer[widx].count.copyToHost(mHMFermatStream));
    CUDA_SAFE_CALL(fermat352.buffer[widx].count.copyToHost(mHMFermatStream));
    CUDA_SAFE_CALL(final.info.copyToHost(mHMFermatStream));
    CUDA_SAFE_CALL(final.count.copyToHost(mHMFermatStream));
    
    // adjust sieves per round
    if (fermat320.buffer[ridx].count[0] && fermat320.buffer[ridx].count[0] < mBlockSize &&
        fermat352.buffer[ridx].count[0] && fermat352.buffer[ridx].count[0] < mBlockSize) {
      mSievePerRound = std::min((unsigned)SW, mSievePerRound+1);
      LOG_F(WARNING, "not enough candidates (%u available, must be more than %u",
             std::max(fermat320.buffer[ridx].count[0], fermat352.buffer[ridx].count[0]),
             mBlockSize);
             
      LOG_F(WARNING, "increase sieves per round to %u", mSievePerRound);
    }
		
		// check candis
		if(candis.size()){
// 			printf("checking %d candis\n", (int)candis.size());
			mpz_class chainorg;
			mpz_class multi;
			for(unsigned i = 0; i < candis.size(); ++i){
				
				fermat_t& candi = candis[i];
				hash_t& hash = hashes.get(candi.hashid);
				
				unsigned age = iteration - hash.iter;
				if(age > PW/2)
          LOG_F(WARNING, "candidate age > PW/2 with %d", age);
				
				multi = candi.index;
				multi <<= candi.origin;
				chainorg = hash.shash;
				chainorg *= multi;
				
				testParams.nCandidateType = candi.type;
        bool isblock = ProbablePrimeChainTestFast(chainorg, testParams, mDepth);
				unsigned chainlength = TargetGetLength(testParams.nChainLength);

				/*printf("candi %d: hashid=%d index=%d origin=%d type=%d length=%d\n",
						i, candi.hashid, candi.index, candi.origin, candi.type, chainlength);*/
				if(chainlength >= block.minshare()){
					
					mpz_class sharemulti = hash.primorial * multi;
					share.set_hash(hash.hash.GetHex());
					share.set_merkle(work.merkle());
					share.set_time(hash.time);
					share.set_bits(work.bits());
					share.set_nonce(hash.nonce);
					share.set_multi(sharemulti.get_str(16));
					share.set_height(block.height());
					share.set_length(chainlength);
					share.set_chaintype(candi.type);
					share.set_isblock(isblock);
					
          LOG_F(1, "GPU %d found share: %d-ch type %d", mID, chainlength, candi.type+1);
					if(isblock)
            LOG_F(1, "GPU %d found BLOCK!", mID);
					
					Send(share, sharepush);
					
				}else if(chainlength < mDepth){
          LOG_F(WARNING, "ProbablePrimeChainTestFast %ubits %d/%d", (unsigned)mpz_sizeinbase(chainorg.get_mpz_t(), 2), chainlength, mDepth);
          LOG_F(WARNING, "origin: %s", chainorg.get_str().c_str());
          LOG_F(WARNING, "type: %u", (unsigned)candi.type);
          LOG_F(WARNING, "multiplier: %u", (unsigned)candi.index);
          LOG_F(WARNING, "layer: %u", (unsigned)candi.origin);
          LOG_F(WARNING, "hash primorial: %s", hash.primorial.get_str().c_str());
          LOG_F(WARNING, "primorial multipliers: ");
          for (unsigned i = 0; i < mPrimorial;) {
            if (hash.primorial % gPrimes[i] == 0) {
              hash.primorial /= gPrimes[i];
              LOG_F(WARNING, " * [%u]%u", i+1, gPrimes[i]);
            } else {
              i++;
            }
          }
					stats.errors++;
				}
			}
		}

		if(MakeExit)
			break;
		
		iteration++;
	}
	
  LOG_F(INFO, "GPU %d stopped.", mID);

	zmq_close(blocksub);
	zmq_close(worksub);
	zmq_close(statspush);
	zmq_close(sharepush);
	czmq_signal(pipe);
}

XPMClient::~XPMClient() {
	
	for(unsigned i = 0; i < mWorkers.size(); ++i)
		if(mWorkers[i].first){
			mWorkers[i].first->MakeExit = true;
      if(czmq_poll(mWorkers[i].second, 8000))
				delete mWorkers[i].first;
		}

	zmq_close(mBlockPub);
	zmq_close(mWorkPub);
	zmq_close(mStatsPull);
}

void XPMClient::dumpSieveConstants(unsigned weaveDepth,
                                   unsigned threadsNum,
                                   unsigned windowSize,
                                   unsigned *primes,
                                   std::ostream &file) 
{
  unsigned ranges[3] = {0, 0, 0};
  for (unsigned i = 0; i < weaveDepth/threadsNum; i++) {
    unsigned prime = primes[i*threadsNum];
    if (ranges[0] == 0 && windowSize/prime <= 2)
      ranges[0] = i;
    if (ranges[1] == 0 && windowSize/prime <= 1)
      ranges[1] = i;
    if (ranges[2] == 0 && windowSize/prime == 0)
      ranges[2] = i;
  }

  file << "#define SIEVERANGE1 " << ranges[0] << "\n";
  file << "#define SIEVERANGE2 " << ranges[1] << "\n";
  file << "#define SIEVERANGE3 " << ranges[2] << "\n";
}

bool XPMClient::Initialize(Configuration* cfg, bool benchmarkOnly, unsigned adjustedKernelTarget) {
	_cfg = cfg;
	
	{
		int np = sizeof(gPrimes)/sizeof(unsigned);
		gPrimes2.resize(np*2);
		for(int i = 0; i < np; ++i){
			unsigned prime = gPrimes[i];
			float fiprime = 1.f / float(prime);
			gPrimes2[i*2] = prime;
			memcpy(&gPrimes2[i*2+1], &fiprime, sizeof(float));
		}
	}

  unsigned clKernelLSize = 1024;
  unsigned clKernelLSizeLog2 = 10;

  std::vector<CUDADeviceInfo> gpus;
  {
    int devicesNum = 0;
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGetCount(&devicesNum));  
    mNumDevices = devicesNum;
  }
  
	int cpuload = cfg->lookupInt("", "cpuload", 1);
	int depth = 5 - cpuload;
	depth = std::max(depth, 2);
	depth = std::min(depth, 5);
	
  onCrash = cfg->lookupString("", "onCrash", "");

	unsigned clKernelTarget = adjustedKernelTarget ? adjustedKernelTarget : 10;
	const char *targetValue = cfg->lookupString("", "target", "auto");
	if (strcmp(targetValue, "auto") != 0) {
		clKernelTarget = atoi(targetValue);
		clKernelTargetAutoAdjust = false;
	}

	bool clKernelWidthAutoAdjust = true;
	unsigned clKernelWidth = clKernelTarget*2;
	const char *widthValue = cfg->lookupString("", "width", "auto");
	if (strcmp(widthValue, "auto") != 0) {
		clKernelWidthAutoAdjust = false;
		clKernelWidth = atoi(widthValue);
	}
	
  unsigned clKernelStripes = cfg->lookupInt("", "sieveSize", 420);
  unsigned clKernelPCount = cfg->lookupInt("", "weaveDepth", 40960);
  unsigned clKernelWindowSize = cfg->lookupInt("", "windowSize", 4096);

	unsigned multiplierSizeLimits[3] = {26, 33, 36};
	std::vector<bool> usegpu(mNumDevices, true);
  std::vector<int> sievePerRound(mNumDevices, 5);
	mCoreFreq = std::vector<int>(mNumDevices, -1);
	mMemFreq = std::vector<int>(mNumDevices, -1);
	mPowertune = std::vector<int>(mNumDevices, 42);
  mFanSpeed = std::vector<int>(mNumDevices, 70);
	
	{
		StringVector cmultiplierlimits;
		StringVector cdevices;
		StringVector csieveperround;
		StringVector ccorespeed;
		StringVector cmemspeed;
		StringVector cpowertune;
		StringVector cfanspeed;
    
		try {
			cfg->lookupList("", "devices", cdevices);
			cfg->lookupList("", "corefreq", ccorespeed);
			cfg->lookupList("", "memfreq", cmemspeed);
			cfg->lookupList("", "powertune", cpowertune);
      cfg->lookupList("", "fanspeed", cfanspeed);
			cfg->lookupList("", "multiplierLimits", cmultiplierlimits);
      cfg->lookupList("", "sievePerRound", csieveperround);
		}catch(const ConfigurationException& ex) {}

		if (cmultiplierlimits.length() == 3) {
			multiplierSizeLimits[0] = atoi(cmultiplierlimits[0]);
			multiplierSizeLimits[1] = atoi(cmultiplierlimits[1]);
			multiplierSizeLimits[2] = atoi(cmultiplierlimits[2]);
		} else {
      LOG_F(WARNING, "invalid multiplierLimits parameter in config, must be list of 3 numbers");
		}
		
		for(int i = 0; i < (int)mNumDevices; ++i){
			
			if(i < cdevices.length())
				usegpu[i] = !strcmp(cdevices[i], "1");
			
			if (i < csieveperround.length())
				sievePerRound[i] = atoi(csieveperround[i]);
			
			if(i < ccorespeed.length())
				mCoreFreq[i] = atoi(ccorespeed[i]);
			
			if(i < cmemspeed.length())
				mMemFreq[i] = atoi(cmemspeed[i]);
			
			if(i < cpowertune.length())
				mPowertune[i] = atoi(cpowertune[i]);
			
      if(i < cfanspeed.length())
        mFanSpeed[i] = atoi(cfanspeed[i]);                        
		}
	}

	for (unsigned i = 0; i < mNumDevices; i++) {
		if (usegpu[i]) {
			char name[128];
			CUDADeviceInfo info;
			mDeviceMap[i] = gpus.size();
			mDeviceMapRev[gpus.size()] = i;
			info.index = i;
			CUDA_SAFE_CALL(cuDeviceGet(&info.device, i));
			CUDA_SAFE_CALL(cuDeviceGetAttribute(&info.majorComputeCapability, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, info.device));
			CUDA_SAFE_CALL(cuDeviceGetAttribute(&info.minorComputeCapability, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, info.device));
			CUDA_SAFE_CALL(cuCtxCreate(&info.context, CU_CTX_SCHED_AUTO, info.device));
			CUDA_SAFE_CALL(cuDeviceGetName(name, sizeof(name), info.device));
			gpus.push_back(info);
      LOG_F(INFO, "[%i] %s; Compute capability %i.%i", (int)gpus.size()-1, name, info.majorComputeCapability, info.minorComputeCapability);
		} else {
			mDeviceMap[i] = -1;
		}
	}
	
	// generate kernel configuration file
  {
    std::ofstream config("xpm/cuda/config.cu", std::fstream::trunc);
    config << "#define STRIPES " << clKernelStripes << '\n';
    config << "#define WIDTH " << clKernelWidth << '\n';
    config << "#define PCOUNT " << clKernelPCount << '\n';
    config << "#define TARGET " << clKernelTarget << '\n';
    config << "#define SIZE " << clKernelWindowSize << '\n';
    config << "#define LSIZE " << clKernelLSize << '\n';
    config << "#define LSIZELOG2 " << clKernelLSizeLog2 << '\n';
    config << "#define LIMIT13 " << multiplierSizeLimits[0] << '\n';
    config << "#define LIMIT14 " << multiplierSizeLimits[1] << '\n';
    config << "#define LIMIT15 " << multiplierSizeLimits[2] << '\n';    
    dumpSieveConstants(clKernelPCount, clKernelLSize, clKernelWindowSize*32, gPrimes+13, config);
  }
  
  std::string arguments = cfg->lookupString("", "compilerFlags", "");

  std::vector<CUmodule> modules;
	modules.resize(gpus.size());
  for (unsigned i = 0; i < gpus.size(); i++) {
		char kernelname[64];
		char ccoption[64];
		sprintf(kernelname, "kernelxpm_gpu%u.ptx", gpus[i].index);
		sprintf(ccoption, "--gpu-architecture=compute_%i%i", gpus[i].majorComputeCapability, gpus[i].minorComputeCapability);
    const char *options[] = { ccoption, arguments.c_str() };
        unsigned noptions = 1;
        if (arguments.size()) {
            ++ noptions;
        }
		CUDA_SAFE_CALL(cuCtxSetCurrent(gpus[i].context));
    if (!cudaCompileKernel(kernelname,
				{ "xpm/cuda/config.cu", "xpm/cuda/procs.cu", "xpm/cuda/fermat.cu", "xpm/cuda/sieve.cu", "xpm/cuda/sha256.cu", "xpm/cuda/benchmarks.cu"},
				options,
        noptions,
				&modules[i],
        gpus[i].majorComputeCapability,
        gpus[i].minorComputeCapability,
				adjustedKernelTarget != 0)) {
			return false;
		}
  }
  
  if (benchmarkOnly) {   
    for (unsigned i = 0; i < gpus.size(); i++) {
      cudaRunBenchmarks(gpus[i].context, gpus[i].device, modules[i], depth, clKernelLSize);
    }
    
    return false;
  } else {
    for(unsigned i = 0; i < gpus.size(); ++i) {
      std::pair<PrimeMiner*,void*> worker;
      PrimeMiner* miner = new PrimeMiner(i, gpus.size(), sievePerRound[i], depth, clKernelLSize);
      miner->Initialize(gpus[i].context, gpus[i].device, modules[i]);
      config_t config = miner->getConfig();
			if ((!clKernelTargetAutoAdjust && config.TARGET != clKernelTarget) ||
					(!clKernelWidthAutoAdjust && config.WIDTH != clKernelWidth) ||
					config.PCOUNT != clKernelPCount ||
					config.STRIPES != clKernelStripes ||
					config.SIZE != clKernelWindowSize ||
					config.LIMIT13 != multiplierSizeLimits[0] ||
					config.LIMIT14 != multiplierSizeLimits[1] ||
					config.LIMIT15 != multiplierSizeLimits[2]) {
        LOG_F(ERROR, "Existing CUDA kernel (kernelxpm_gpu<N>.ptx) incompatible with configuration");
        LOG_F(ERROR, "Please remove kernelxpm_gpu<N>.ptx file and restart miner");
        exit(1);
      }

      void *pipe = czmq_thread_fork(mCtx, &PrimeMiner::InvokeMining, miner);
      czmq_wait(pipe);
      czmq_signal(pipe);
      worker.first = miner;
      worker.second = pipe;    
      mWorkers.push_back(worker);
    }
  }
	  
	return true;
}


void XPMClient::NotifyBlock(const proto::Block& block) {
	
	SendPub(block, mBlockPub);
	
}


bool XPMClient::TakeWork(const proto::Work& work) {
	
	const double TargetIncrease = 0.994;
	const double TargetDecrease = 0.0061;
	
	SendPub(work, mWorkPub);
	
	if (!clKernelTargetAutoAdjust || work.bits() == 0)
		return true;
	double difficulty = GetPrimeDifficulty(work.bits());

	bool needReset = false;
	for(unsigned i = 0; i < mWorkers.size(); ++i) {
		PrimeMiner *miner = mWorkers[i].first;
		double target = miner->getConfig().TARGET;
		if (difficulty > target && difficulty-target >= TargetIncrease) {
      LOG_F(WARNING, "Target with high difficulty detected, need increase miner target");
			needReset = true;
			break;
		} else if (difficulty < target && target-difficulty >= TargetDecrease) {
      LOG_F(WARNING, "Target with low difficulty detected, need decrease miner target");
			needReset = true;
			break;
		}
	}
	
	if (needReset) {
		unsigned newTarget = TargetGetLength(work.bits());
		if (difficulty - newTarget >= TargetIncrease)
			newTarget++;
    LOG_F(WARNING, "Rebuild miner kernels, adjust target to %u..", newTarget);
		// Stop and destroy all workers
		for(unsigned i = 0; i < mWorkers.size(); ++i) {
      LOG_F(WARNING, "attempt to stop GPU %u ...", i);
			if(mWorkers[i].first){
				mWorkers[i].first->MakeExit = true;
        if(czmq_poll(mWorkers[i].second, 8000)) {
					delete mWorkers[i].first;
          zmq_close(mWorkers[i].second);
        }
			}
		}
		
		mWorkers.clear();
		
		// Build new kernels with adjusted target
		mPaused = true;
		Initialize(_cfg, false, newTarget);
    Toggle();
		return false;
	} else {
		return true;
	}
}


int XPMClient::GetStats(proto::ClientStats& stats) {
	
	unsigned nw = mWorkers.size();
	std::vector<bool> running(nw);
	std::vector<stats_t> wstats(nw);
	
	while (czmq_poll(mStatsPull, 0)) {
		zmq_msg_t msg;
		zmq_msg_init(&msg);
		zmq_recvmsg(mStatsPull, &msg, 0);          
		size_t fsize = zmq_msg_size(&msg);
		uint8_t *fbytes = (uint8_t*)zmq_msg_data(&msg);
		if(fsize >= sizeof(stats_t)) {
			stats_t* tmp = (stats_t*)fbytes;
			if(tmp->id < nw){
				running[tmp->id] = true;
				wstats[tmp->id] = *tmp;
			}
		}
		
		zmq_msg_close(&msg); 
	}

	double cpd = 0;
	unsigned errors = 0;
	int maxtemp = 0;
	unsigned ngpus = 0;
  int crashed = 0;
  
	for(unsigned i = 0; i < nw; ++i){
		
		int devid = mDeviceMapRev[i];
		int temp = gpu_temp(devid);
		int activity = gpu_activity(devid);
		
		if(temp > maxtemp)
			maxtemp = temp;
		
		cpd += wstats[i].cpd;
		errors += wstats[i].errors;
		
		if(running[i]){
			ngpus++;
      LOG_F(INFO, "[GPU %d] T=%dC A=%d%% E=%d primes=%f fermat=%d/sec cpd=%.2f/day",
					i, temp, activity, wstats[i].errors, wstats[i].primeprob, wstats[i].fps, wstats[i].cpd);
		}else if(!mWorkers[i].first)
      LOG_F(ERROR, "[GPU %d] failed to start!\n", i);
		else if(mPaused) {
      LOG_F(INFO, "[GPU %d] paused", i);
    } else {
      crashed++;
      LOG_F(ERROR, "[GPU %d] crashed!", i);
    }
		
	}
	
  if (crashed && onCrash[0] != '\0') {
    LOG_F(INFO, "Run command %s", onCrash);
    loguru::flush();
    system(onCrash);
  }
	
	if(mStatCounter % 10 == 0)
		for(unsigned i = 0; i < mNumDevices; ++i){
			int gpuid = mDeviceMap[i];
			if(gpuid >= 0)
        LOG_F(INFO, "GPU %d: core=%dMHz mem=%dMHz powertune=%d fanspeed=%d",
						gpuid, gpu_engineclock(i), gpu_memclock(i), gpu_powertune(i), gpu_fanspeed(i));
		}
	
	stats.set_cpd(cpd);
	stats.set_errors(errors);
	stats.set_temp(maxtemp);
	
	mStatCounter++;
	
	return ngpus;
	
}


void XPMClient::Toggle()
{
  for(unsigned i = 0; i < mWorkers.size(); ++i) {
    if(mWorkers[i].first)
      czmq_signal(mWorkers[i].second);
  }

  mPaused = !mPaused;
}


void XPMClient::setup_adl()
{
}
