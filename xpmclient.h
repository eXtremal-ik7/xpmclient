/*
 * xpmclient.h
 *
 *  Created on: 01.05.2014
 *      Author: mad
 */

#ifndef XPMCLIENT_H_
#define XPMCLIENT_H_


#include <gmp.h>
#include <gmpxx.h>

#include "baseclient.h"
#include "opencl.h"
#include "uint256.h"
#include "sha256.h"

#define FERMAT_PIPELINES 2

#define PW 512        // Pipeline width (number of hashes to store)
#define SW 16         // maximum number of sieves in one iteration
#define MSO 128*1024  // max sieve output
#define MFS 2*SW*MSO  // max fermat size

const unsigned maxHashPrimorial = 16;

extern unsigned gPrimes[96*1024];
extern std::vector<unsigned> gPrimes2;

extern cl_program gProgram;





struct stats_t {
	
	unsigned id;
	unsigned errors;
	unsigned fps;
	double primeprob;
	double cpd;
	
	stats_t(){
		id = 0;
		errors = 0;
		fps = 0;
		primeprob = 0;
		cpd = 0;
	}
	
};


struct config_t {
	
	cl_uint N;
	cl_uint SIZE;
	cl_uint STRIPES;
	cl_uint WIDTH;
	cl_uint PCOUNT;
	cl_uint TARGET;
  cl_uint WINDOWSIZE_;
};


template<typename T> class lifoBuffer {
private:
  T *_data;
  size_t _size;
  size_t _readPos;
  size_t _writePos;
  
  size_t nextPos(size_t pos) { return (pos+1) % _size; }
  
public:
  lifoBuffer(size_t size) : _size(size), _readPos(0), _writePos(0) {
    _data = new T[size];
  }
  
  size_t readPos() const { return _readPos; }
  size_t writePos() const { return _writePos; }
  T *data() const { return _data; }
  T& get(size_t index) const { return _data[index]; }
  
  bool empty() const {
    return _readPos == _writePos;
  }  
  
  size_t remaining() const {
    return _writePos >= _readPos ?
    _writePos - _readPos :
    _size - (_readPos - _writePos);
  }
  
  void clear() {
    _readPos = _writePos;
  }
  
  size_t push(const T& element) {
    size_t oldWritePos = _writePos;
    size_t nextWritePos = nextPos(_writePos);
    if (nextWritePos != _readPos) {
      _data[_writePos] = element;
      _writePos = nextWritePos;
    }
    
    return oldWritePos;
  }
  
  size_t pop() {
    size_t oldReadPos = _readPos;
    if (!empty())
      _readPos = nextPos(_readPos);
    return oldReadPos;
  }
};

class PrimeMiner {
public:
	
	struct block_t {
		
		static const int CURRENT_VERSION = 2;
		
		int version;
		uint256 hashPrevBlock;
		uint256 hashMerkleRoot;
		unsigned int time;
		unsigned int bits;
		unsigned int nonce;
		
	};
	
	struct search_t {
		
		clBuffer<cl_uint> midstate;
		clBuffer<cl_uint> found;
    clBuffer<cl_uint> primorialBitField;
		clBuffer<cl_uint> count;
		
	};
	
	struct hash_t {
		
		unsigned iter;
		unsigned nonce;
		unsigned time;
		uint256 hash;
		mpz_class shash;
    mpz_class primorial;
    unsigned primorialIdx;
	};
	
	struct fermat_t {
		cl_uint index;
    cl_uint hashid;
		cl_uchar origin;
		cl_uchar chainpos;
		cl_uchar type;
		cl_uchar reserved;
	};
	
	struct info_t {
		
		clBuffer<fermat_t> info;
		clBuffer<cl_uint> count;
		
	};
	
	struct pipeline_t {
		unsigned current;
		unsigned bsize;
		clBuffer<cl_uint> input;
		clBuffer<cl_uchar> output;
		info_t buffer[2];
	};
  
  struct sieve_t {
    info_t cunningham1[1];
    info_t cunningham2[1];
  };
	
	
  PrimeMiner(unsigned id, unsigned threads, unsigned hashprim, unsigned prim, unsigned sievePerRound, unsigned depth, unsigned LSize);
	~PrimeMiner();
	
	bool Initialize(cl_device_id dev);
	
	static void InvokeMining(void *args, zctx_t *ctx, void *pipe);
  config_t getConfig() { return mConfig; }
	
	bool MakeExit;
	
private:
  void FermatInit(pipeline_t &fermat, unsigned mfs);
  
  void FermatDispatch(pipeline_t &fermat,
                      clBuffer<fermat_t>  sieveBuffers[SW][FERMAT_PIPELINES][2],
                      clBuffer<cl_uint> candidatesCountBuffers[SW][2],
                      unsigned pipelineIdx,
                      int ridx,
                      int widx,
                      uint64_t &testCount,
                      uint64_t &fermatCount,
                      cl_kernel fermatKernel,
                      unsigned sievePerRound);
  
	void Mining(zctx_t *ctx, void *pipe);
	
	unsigned mID;
	unsigned mThreads;
	
	config_t mConfig;
	unsigned mPrimorial;
  unsigned mSievePerRound;
	unsigned mHashPrimorial;
	unsigned mBlockSize;
	cl_uint mDepth;
  unsigned mLSize;  
	
	cl_command_queue mSmall;
	cl_command_queue mBig;
	
	cl_kernel mHashMod;
	cl_kernel mSieveSetup;
	cl_kernel mSieve;
	cl_kernel mSieveSearch;
	cl_kernel mFermatSetup;
	cl_kernel mFermatKernel352;
  cl_kernel mFermatKernel320;  
	cl_kernel mFermatCheck;
	
  info_t final;	
};




class XPMClient {
public:
	
	XPMClient(zctx_t* ctx);
	~XPMClient();
  
	bool Initialize(Configuration* cfg, bool benchmarkOnly = false);
	
	void NotifyBlock(const proto::Block& block);
	
	void TakeWork(const proto::Work& work);
	
	int GetStats(proto::ClientStats& stats);
	
	void Toggle();
	
	void setup_adl();
	
private:
  enum PlatformType {
    ptAMD = 0,
    ptNVidia,
    ptIntel
  };
  
  PlatformType platformType;
  
	zctx_t* mCtx;
	
	std::vector<std::pair<PrimeMiner*, void*> > mWorkers;
	std::map<int,int> mDeviceMap;
	std::map<int,int> mDeviceMapRev;
	
	void* mBlockPub;
	void* mWorkPub;
	void* mStatsPull;
	
	unsigned mNumDevices;
	unsigned mStatCounter;
	bool mPaused;
	
	std::vector<int> mCoreFreq;
	std::vector<int> mMemFreq;
	std::vector<int> mPowertune;
  std::vector<int> mFanSpeed;
   
  int exitType; 
	
  void dumpSieveConstants(unsigned weaveDepth,
                          unsigned threadsNum,
                          unsigned windowSize,
                          unsigned *primes,
                          std::ostream &file);
	
};



extern XPMClient* gClient;













#endif /* XPMCLIENT_H_ */
