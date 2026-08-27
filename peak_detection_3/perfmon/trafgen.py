#!/usr/bin/python3

import os
import sys
import json
from argparse import ArgumentParser

# Parameters - make these command-line arguments later

parser = ArgumentParser ("Generate traffic between node pairs as specified in a JSON config file (tests.json by default)")
parser.add_argument ('-c', '--config', default = "tests.json", help = "Specify JSON config file (see tests.json example)")
args = parser.parse_args ()

with open (args.config) as f:
	tests = json.load (f)
	
pids = []
cwd = os.getcwd()

for test in tests:
	print ('Starting test: %s' % test['testname']); 
	
	if 'host1' in test:

# Start host1 process (generally server) first
		pid = os.fork()

		if pid:
			pids.append ({
				'pid': pid,
				'host': test['host1']['name']
			})
		else:
			os.system ('ssh -o StrictHostKeyChecking=no root@%s "PATH=$PATH:%s sleep %i ; %s 2>&1 > /dev/null"' % (test['host1']['name'], cwd, test['host1']['start_time'], test['host1']['cmd']))
			exit (0)

# Start host2 process 

	if 'host2' in test:
		pid = os.fork()

		if pid:
			pids.append ({
				'pid': pid,
				'host': test['host2']['name']
			})
		else:
			os.system ('ssh -o StrictHostKeyChecking=no root@%s "PATH=$PATH:%s sleep %i ; %s 2>&1 > /dev/null"' % (test['host2']['name'], cwd, test['host2']['start_time'], test['host2']['cmd']))
			exit (0)

print ('Waiting for pending tasks to complete')

for pid in pids:
	try:
		os.waitpid (-1, 0)
		print ('Completed on host %s' % (pid['host']), file=sys.stderr)
	except ChildProcessError:
		print ('All done', file=sys.stderr)
