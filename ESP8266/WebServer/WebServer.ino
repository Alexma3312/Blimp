#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

ESP8266WebServer server;
uint8_t pin_led = 16;
uint8_t pin_motor = 15;
char* ssid = "OPPO Find X";
char* password = "12345678";

void toggleLED()
{
  digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(1000);                       // wait for a second
  digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
  delay(1000);  
  digitalWrite(pin_motor, 1.5); 
  delay(1000);
  digitalWrite(pin_motor, -1.5); 
  delay(1000);
  server.send(204,"");
}


void setup() {
  pinMode(pin_led, OUTPUT);
  pinMode(pin_motor, OUTPUT);
//  WiFi.begin(ssid, password);
//  Serial.begin(115200);
//  while(WiFi.status()!=WL_CONNECTED)
//  {
//    Serial.print(".");
//    delay(500);
//  }
//  Serial.println("");
//  Serial.println("WiFi connected");
//  Serial.println("IP address: ");
//  Serial.println(WiFi.localIP());
//
//  server.on("/",[](){server.send(200,"text/plain","Hello World");});
//  
//  server.begin();
  
}

void loop() {
  digitalWrite(pin_led, 1.5); 
  delay(1000);
  digitalWrite(pin_led, -1.5); 
  delay(1000);
//  server.handleClient();
//  server.on("/toggle",toggleLED);

}
