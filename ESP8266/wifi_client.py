import httplib2
import keyboard

http = httplib2.Http()

url = "http://192.168.43.195/toggle"
response, content = http.request(url,"Get")
print(response)
