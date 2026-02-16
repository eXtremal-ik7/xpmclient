/*
 * sysinfo.h
 *
 *  Created on: 30.04.2014
 *      Author: mad
 */

#ifndef SYSINFO_H_
#define SYSINFO_H_


#ifdef WIN32
#include <windows.h>
#else
#include <sys/utsname.h>
#endif





namespace sysinfo {


static std::string GetClientName() {
	
	std::string name;
	
#ifdef WIN32
	TCHAR computerName[MAX_COMPUTERNAME_LENGTH + 1];
	DWORD size = sizeof(computerName) / sizeof(computerName[0]);
	GetComputerName(computerName, &size);
	name = std::string(computerName, size);
#else
	static struct utsname u;  
	if(uname(&u) < 0)
		name = "unknown";
	name = std::string(u.nodename);
#endif
	
	return name;
	
}

static void getCpuid(unsigned int* p, unsigned int ax)
{
	__asm __volatile
	(   "movq %%rbx, %%rsi\n\t"
		"cpuid\n\t"
		"xchgq %%rbx, %%rsi"
		: "=a" (p[0]), "=S" (p[1]),
		  "=c" (p[2]), "=d" (p[3])
		: "0" (ax)
	);
}

static unsigned GetClientID() {
	
	unsigned id = 0;
	
#ifdef WIN32
	DWORD serialNum = 0;
	GetVolumeInformation("c:\\", NULL, 0, &serialNum, NULL, NULL, NULL, 0);
	id = serialNum;
#else
	unsigned int cpuinfo[4] = { 0, 0, 0, 0 };          
  getCpuid(cpuinfo, 0);
	id = cpuinfo[0] * cpuinfo[1] * cpuinfo[2] * cpuinfo[3];
#endif
	
	return id;
	
}

}; // namespace sysinfo






#endif /* SYSINFO_H_ */
