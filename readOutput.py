
import sys

def main():
	res = []
	resultTable = ''
	for line in sys.stdin:
		words = line.split('=')
		schedule = 'Affinity'
		if words[0].strip() == 'Threads':
			numProcs = words[1].strip()
		if 'loop 1' in words[0]:
			loop1time = words[1].strip()
		if 'loop 2' in words[0]:
			loop2time = words[1].strip()
			res.append([schedule,numProcs,loop1time, loop2time])

	for row in res:
		for item in row:
			resultTable+= item+','
		resultTable+='\n'
 
	print resultTable

if __name__=='__main__':
	main()
