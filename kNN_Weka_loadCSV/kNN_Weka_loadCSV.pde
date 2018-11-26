//*********************************************
// Weka for Processing
// KNN_LoadModel
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

Table csvData;
String fileName = "data/testData.csv";
boolean b_saveCSV = false;
boolean b_train = false;
boolean b_test = false;
boolean b_showData = true;

int label = 0;
PGraphics pg;
int K = 1;

void setup() {
  size(500, 500);

  //Initiate the dataList and set the header of table
  csvData = loadTable("testData.csv", "header");

  pg = createGraphics(width, height);
}

void draw() {
  background(255);
  if (b_saveCSV) {
    saveTable(csvData, fileName); //Save the current table as a CSV file to the file folder 
    println("Saved as: ", fileName);
    b_saveCSV = false; //reset b_saveCSV;
  }

  if (b_train) {
    try {
      initTrainingSet(csvData); // in Weka.pde
      cls = new IBk(K); //IBk(int k): kNN classifier.
      cls.buildClassifier(training); //Train the classifier
      println(cls);
      weka.core.SerializationHelper.write(dataPath("kNN.model"), cls);
      if (K==1) pg = getModelImage(pg, (Classifier)cls, training); 
      else pg = createGraphics(width, height); // cannot show the KNN model image for now

      printEvalResults("testData.csv", 2, 10);

      b_train = false;
      b_test = true;
    } 
    catch (Exception e) {
      e.printStackTrace();
    }
  }

  image(pg, 0, 0);
  pushStyle();
  fill(0);
  textSize(24);
  text("K="+K+", Current Label="+label, 20, 20);
  popStyle();
  
  for (int i = 0; i < csvData.getRowCount(); i++) { 
    TableRow row = csvData.getRow(i);
    float x = row.getFloat("x");
    float y = row.getFloat("y");
    float id = row.getFloat("label");
    int index = (int) id;

    float[] features = { x, y }; //form a feature array
    drawDataPoint(index, features); //draw the data on the Canvas
  }

  if (b_test) {
    Instance inst = new DenseInstance(3);     
    inst.setValue(training.attribute(0), (float)mouseX); 
    inst.setValue(training.attribute(1), (float)mouseY); 

    // "instance" has to be associated with "Instances"
    Instances testData = new Instances("Test Data", attributes, 0);
    testData.add(inst);
    testData.setClassIndex(2);        

    float classification = -1;
    try {
      classification = (float) cls.classifyInstance(testData.firstInstance());
    } 
    catch (Exception e) {
      e.printStackTrace();
    } 
    drawMouseCursor((int)classification);
  } else {
    drawMouseCursor(label);
  }
}

void mousePressed() {
  if (mouseButton == RIGHT) {
    ++label;
    label %= 10;
  }
}

void mouseDragged() {
  //add a row with new data 
  TableRow newRow = csvData.addRow();
  newRow.setFloat("x", mouseX);
  newRow.setFloat("y", mouseY);
  newRow.setFloat("label", label);
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
  if (key == 'D' || key == 'd') {
    b_showData = (b_showData? false: true);
  }
  if (key == ' ') {
    csvData.clearRows();
    pg = createGraphics(width, height);
    label = 0;
  }
  if (key >= '0' && key <= '9') {
    label = key - '0';
  }
  if (key == 'A' || key == 'a') {
    K+=2;
    K%=8;
    b_train = true;
    b_test = false;
    b_saveCSV = true;
  }
}

void drawMouseCursor(int _index) {
  color colors[] = {
    color(155, 89, 182), color(63, 195, 128), color(214, 69, 65), 
    color(82, 179, 217), color(244, 208, 63), color(242, 121, 53), 
    color(0, 121, 53), color(128, 128, 0), color(52, 0, 128), 
    color(128, 52, 0), color(52, 128, 0), color(128, 52, 0)
  };
  pushStyle();
  textSize(12);
  textAlign(CENTER, CENTER);
  if (mousePressed) {
    stroke(0);
    fill(255);
  } else { 
    noStroke();
    fill(colors[_index]);
  }
  ellipse(mouseX, mouseY, 20, 20);

  if (mousePressed) {
    noStroke();
    fill(0);
  } else { 
    fill(255);
  }
  text(_index, mouseX, mouseY);

  popStyle();
}

void drawDataPoint(int _index, float[] _features) {
  color colors[] = {
    color(155, 89, 182), color(63, 195, 128), color(214, 69, 65), 
    color(82, 179, 217), color(244, 208, 63), color(242, 121, 53), 
    color(0, 121, 53), color(128, 128, 0), color(52, 0, 128), 
    color(128, 52, 0), color(52, 128, 0), color(128, 52, 0)
  };
  pushStyle();
  textSize(12);
  textAlign(CENTER, CENTER);

  stroke(0);
  fill(colors[_index]);
  ellipse(_features[0], _features[1], 20, 20);

  noStroke();
  fill(0);
  text(_index, _features[0], _features[1]);

  popStyle();
}