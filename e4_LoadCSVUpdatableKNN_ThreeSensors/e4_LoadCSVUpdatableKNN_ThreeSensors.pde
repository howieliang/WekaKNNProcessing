//*********************************************
// CSV Processing
// e1_kNN_ThreeSensors
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************
// S for saving the file
// [SPACE] for refreshing the data

import processing.serial.*;
Serial port; 

int[] rawData;
int sensorNum = 3; 
int dataNum = 500;

Table csvData;
String fileName = "data/testData.csv";
boolean b_saveCSV = false;
boolean b_train = false;
boolean b_test = false;

int label = 0;
int dataCnt = 0;

PGraphics pg;
int K = 1;
int[] testFeatures; 

void setup() {
  size(500, 500);

  //Initiate the dataList and set the header of table
  csvData = loadTable("testData.csv", "header");
  dataNum = csvData.getRowCount();
  dataCnt = csvData.getRowCount();
  //add more columns here

  //Initiate the serial port
  rawData = new int[sensorNum];
  for (int i = 0; i < Serial.list().length; i++) println("[", i, "]:", Serial.list()[i]);
  String portName = Serial.list()[Serial.list().length-1];//MAC: check the printed list
  //String portName = Serial.list()[9];//WINDOWS: check the printed list
  port = new Serial(this, portName, 115200);
  port.bufferUntil('\n'); // arduino ends each data packet with a carriage return 
  port.clear();           // flush the Serial buffer

  pg = createGraphics(width, height);
  testFeatures = new int[sensorNum];
}

void draw() {
  background(255);

  if (b_saveCSV) {
    //Save the table to the file folder
    saveTable(csvData, fileName); //save table as CSV file
    println("Saved as: ", fileName);

    //reset b_saveCSV;
    b_saveCSV = false;
  }

  if (b_train) {
    try {
      initTrainingSet(csvData); // in Weka.pde
      cls = new IBk(K); //IBk(int k): kNN classifier.
      cls.buildClassifier(training); //Train the classifier
      println(cls);
      weka.core.SerializationHelper.write(dataPath("kNN.model"), cls);
      //if (K==1) pg = getModelImage(pg, (Classifier)cls, training); 
      //else pg = createGraphics(width, height); // cannot show the KNN model image for now

      printEvalResults("testData.csv", sensorNum, 10);

      b_train = false;
      b_test = true;
    } 
    catch (Exception e) {
      e.printStackTrace();
    }
  }
  //image(pg, 0, 0);

  if (b_test) {
    Instance inst = new DenseInstance(sensorNum+1);     
    inst.setValue(training.attribute(0), (float)testFeatures[0]); 
    inst.setValue(training.attribute(1), (float)testFeatures[1]);
    inst.setValue(training.attribute(2), (float)testFeatures[2]);

    // "instance" has to be associated with "Instances"
    Instances testData = new Instances("Test Data", attributes, 0);
    testData.add(inst);
    testData.setClassIndex(sensorNum);        

    float classification = -1;
    try {
      classification = (float) cls.classifyInstance(testData.firstInstance());
    } 
    catch (Exception e) {
      e.printStackTrace();
    } 

    String result = "y="+(int)classification+", X=["+(float)testFeatures[0]+","+(float)testFeatures[1]+","+(float)testFeatures[2]+"]";
    pushStyle();
    fill(0);
    textSize(24);
    if (csvData.getRowCount()==dataCnt) text(result, 20, 20);
    else text("Update: Label "+label+", "+csvData.getRowCount()+"/"+dataNum, 20, 20);
    popStyle();
  }
  for (int i = 0; i < csvData.getRowCount(); i++) { 
    //read the values from the file
    TableRow row = csvData.getRow(i);
    float x = row.getFloat("x");
    float y = row.getFloat("y");
    float z = row.getFloat("z");
    // add more features here if you have

    //form a feature array
    float[] features = { x, y, z }; //form an array of input features

    //draw the data on the Canvas: 
    //Note: the row index is used as the label instead
    drawDataPoint1D(i, features);
  }
}

void serialEvent(Serial port) {   
  String inData = port.readStringUntil('\n');  // read the serial string until seeing a carriage return
  if (inData.charAt(0) == 'A') {  
    rawData[0] = int(trim(inData.substring(1)));
    return;
  }
  if (inData.charAt(0) == 'B') {  
    rawData[1] = int(trim(inData.substring(1)));
    return;
  }
  if (inData.charAt(0) == 'C') {  
    rawData[2] = int(trim(inData.substring(1)));
    //add a new row of data
    boolean updated = false;
    if (csvData.getRowCount() < dataCnt) {
      //add a row with new data 
      TableRow newRow = csvData.addRow();
      newRow.setFloat("x", rawData[0]);
      newRow.setFloat("y", rawData[1]);
      newRow.setFloat("z", rawData[2]);
      newRow.setFloat("index", label);
      updated = true;
    }
    testFeatures[0] = rawData[0];
    testFeatures[1] = rawData[1];
    testFeatures[2] = rawData[2];

    if (updated) {
      Instance inst = new DenseInstance(sensorNum+1);     
      inst.setValue(training.attribute(0), (float)rawData[0]); 
      inst.setValue(training.attribute(1), (float)rawData[1]);
      inst.setValue(training.attribute(2), (float)rawData[2]);
      inst.setValue(training.attribute(3), label);

      // "instance" has to be associated with "Instances"
      Instances testData = new Instances("Test Data", attributes, 0);
      testData.add(inst);
      testData.setClassIndex(sensorNum);        
      try {
        cls.updateClassifier(testData.firstInstance());
      } 
      catch (Exception e) {
        e.printStackTrace();
      }
      updated = false;
    }
    return;
  }
}

void keyPressed() {
  if (key == 'S' || key == 's') {
    b_saveCSV = true;
  }
  if (key == 'T' || key == 't') {
    b_train = true;
    b_test = false;
    b_saveCSV = true;
  }
  if (key == ' ') {
    csvData.clearRows();
  }
  if (key >= '0' && key <= '9') {
    label = key - '0';
    if (dataCnt<dataNum) dataCnt+=50;
    if (b_test) {
      dataNum+=50;
      dataCnt+=50;
    }
  }
}

//functions for drawing the data
void drawDataPoint1D(int _i, float[] _features) { 
  float pD = max(width/dataNum,1);
  float pX = map(((float)_i+0.5)/(float)dataNum, 0, 1, 0, width);
  float[] pY = new float[_features.length];
  for (int j = 0; j < _features.length; j++) pY[j] = map(_features[j], 0, 1024, 0, height) ; 
  pushStyle();
  for (int j = 0; j < _features.length; j++) {
    noStroke();
    if (j==0)fill(255, 0, 0);
    else if (j==1)fill(0, 255, 0);
    else if (j==2)fill(0, 0, 255);
    else if (j==3)fill(255, 0, 255);
    else fill(0);
    ellipse(pX, pY[j], pD, pD);
  }
  popStyle();
}