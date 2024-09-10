import speedtest
test = speedtest.Speedtest()
print('Download: ', test.download())
print('Upload: ', test.upload())
print('Ping: ', test.results.ping)