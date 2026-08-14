import ssfileparser as ss
ssfile = 'binary_examples.ss'
print('signature count:' +str(ss.get_signature_count(ssfile)))
sssig = ss.get_signature(ssfile,0)
print(sssig['data'][0])