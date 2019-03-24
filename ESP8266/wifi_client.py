import httplib2
import keyboard

http = httplib2.Http()

url = "http://192.168.43.195/toggle"
while(1):
    response, content = http.request(url,"Get")
    print(response)
