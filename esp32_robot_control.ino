#include <WiFi.h>
#include <WiFiServer.h>

// WiFi credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Motor driver pins (L298N)
const int motor1Pin1 = 2;  // IN1
const int motor1Pin2 = 4;  // IN2
const int motor2Pin1 = 16; // IN3
const int motor2Pin2 = 17; // IN4
const int enable1Pin = 18; // ENA
const int enable2Pin = 19; // ENB

// PWM properties
const int freq = 30000;
const int pwmChannel1 = 0;
const int pwmChannel2 = 1;
const int resolution = 8;

// Speed settings
int normalSpeed = 150;  // Normal speed (0-255)
int fastSpeed = 255;    // Fast speed (0-255)
int currentSpeed = normalSpeed;

// Server setup
WiFiServer server(80);

void setup() {
  Serial.begin(115200);
  
  // Initialize motor pins
  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(motor2Pin1, OUTPUT);
  pinMode(motor2Pin2, OUTPUT);
  pinMode(enable1Pin, OUTPUT);
  pinMode(enable2Pin, OUTPUT);
  
  // Configure PWM
  ledcSetup(pwmChannel1, freq, resolution);
  ledcSetup(pwmChannel2, freq, resolution);
  ledcAttachPin(enable1Pin, pwmChannel1);
  ledcAttachPin(enable2Pin, pwmChannel2);
  
  // Stop motors initially
  stopMotors();
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println();
  Serial.println("WiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  
  // Start server
  server.begin();
  Serial.println("Server started");
}

void loop() {
  WiFiClient client = server.available();
  
  if (client) {
    Serial.println("New client connected");
    
    while (client.connected()) {
      if (client.available()) {
        String command = client.readString();
        command.trim();
        
        Serial.print("Received command: ");
        Serial.println(command);
        
        // Process command
        processCommand(command);
      }
    }
    
    client.stop();
    Serial.println("Client disconnected");
  }
}

void processCommand(String command) {
  if (command == "forward") {
    moveForward();
  } else if (command == "backward") {
    moveBackward();
  } else if (command == "left") {
    turnLeft();
  } else if (command == "right") {
    turnRight();
  } else if (command == "stop") {
    stopMotors();
  } else if (command == "fast") {
    currentSpeed = fastSpeed;
    Serial.println("Fast mode activated");
  } else {
    stopMotors();
    Serial.println("Unknown command, stopping");
  }
}

void moveForward() {
  Serial.println("Moving forward");
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);
  
  ledcWrite(pwmChannel1, currentSpeed);
  ledcWrite(pwmChannel2, currentSpeed);
}

void moveBackward() {
  Serial.println("Moving backward");
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, HIGH);
  
  ledcWrite(pwmChannel1, currentSpeed);
  ledcWrite(pwmChannel2, currentSpeed);
}

void turnLeft() {
  Serial.println("Turning left");
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);
  
  ledcWrite(pwmChannel1, currentSpeed);
  ledcWrite(pwmChannel2, currentSpeed);
}

void turnRight() {
  Serial.println("Turning right");
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, HIGH);
  
  ledcWrite(pwmChannel1, currentSpeed);
  ledcWrite(pwmChannel2, currentSpeed);
}

void stopMotors() {
  Serial.println("Stopping motors");
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, LOW);
  
  ledcWrite(pwmChannel1, 0);
  ledcWrite(pwmChannel2, 0);
  
  // Reset to normal speed when stopping
  currentSpeed = normalSpeed;
}

