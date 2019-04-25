#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

ESP8266WebServer server;
uint8_t pin_led = 16;
uint8_t pin_motor_1 = 5;
uint8_t pin_motor_2 = 4;
char* ssid = "OPPO Find X";
char* password = "12345678";

void toggleLED()
{
//  digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on (HIGH is the voltage level)
//  delay(1000);                       // wait for a second
//  digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
//  delay(1000);  
//  digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on (HIGH is the voltage level)
//  delay(1000);                       // wait for a second
//  digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
//  delay(1000);  
  digitalWrite(pin_motor_1, HIGH); 
  digitalWrite(pin_motor_2, LOW); 
  
//  delay(100);
  server.send(204,"Message Receive");
}


void setup() {
  pinMode(pin_led, OUTPUT);
  pinMode(pin_motor_1, OUTPUT);  
  pinMode(pin_motor_2, OUTPUT);
  WiFi.begin(ssid, password);
  Serial.begin(115200);
  while(WiFi.status()!=WL_CONNECTED)
  {
    Serial.print(".");
    delay(500);
  }
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  server.on("/",[](){server.send(200,"text/plain","Hello World");});
  
  server.begin();
  
}

void loop() {
  digitalWrite(pin_led, HIGH); 
  delay(1000);
  server.handleClient();
  server.on("/toggle",toggleLED);

}
