#include "filereader.h"

FileReader::FileReader() {
  check = static_cast<char*>(malloc(sizeof(char) * 4));
  firstImageRead = true;
}

FileReader::~FileReader() {
  free(imgBuffer);
  free(check);
}

// Read in a bitmap file and stores the actual data into a buffer
// Returns false if the bitmap file is the wrong type of bitmap
bool FileReader::readBitmap(int fileNum) {
  int dataOffset, dataSize;
  std::string fileName;

  std::ostringstream stringstream;
  stringstream << fileNum;
  std::string fileNumber = stringstream.str();

  fileName = "img" + fileNumber + ".bmp";

  fs.open(fileName.c_str(), std::ios::in | std::ios::binary);

  fs.read(check, 2);

  // Check if the file is indeed a bitmap file
  if (*check != 'B' && *(check + 1) != 'M') {
    printf("Image %d is not a bitmap file\n\r", fileNum);
    return false;
  }
  fs.seekg(BMPBPP);
  fs.read(check, 2);

  // Check if the image is monochrome
  if (static_cast<int>(*check) != 1) {
    printf("Image %d is not a monochrome image\n\r", fileNum);
    return false;
  }

  // Get data offset
  fs.seekg(DATAOFFSET);
  fs.read(check, 4);

  dataOffset = bytesToInt(check, 4);

  // Get the data size in bytes
  fs.seekg(DATASIZEINBYTES);
  fs.read(check, 4);

  dataSize = bytesToInt(check, 4);
  imageSize = dataSize;

  fs.seekg(COMP);
  fs.read(check, 4);

  // If this is the first image we read
  if (firstImageRead) {
    // allocate image buffer
    imgBuffer = static_cast<char*>(malloc(dataSize));
    // make sure it is not re-allocated
    firstImageRead = false;
  } else {
    fs.seekg(BMPWIDTH);
    fs.read(check, 2);
    if (bytesToInt(check, 2) != width) {
      printf("Image %d does not have the same width as the initializing image\n\r", fileNum);
      return false;
    }

    fs.seekg(BMPHEIGHT);
    fs.read(check, 2);
    if (bytesToInt(check, 2) != height) {
      printf("Image %d does not have the same height as the initializing image\n\r", fileNum);
      return false;
    }
  }

  // Get the actual image data
  fs.seekg(dataOffset);
  fs.read(imgBuffer, dataSize);
  fs.close();
  return true;
}

// Read in the first image to grab the dimensions for the rest of the training data
int FileReader::getBitmapDimensions() {
  std::ifstream stream("img0.bmp", std::ios::in | std::ios::binary);
  stream.read(check, 2);

  // Check to make sure it is a bitmap file
  if (*check != 'B' && *(check + 1) != 'M')
    return -1;

  // Grab the width of the image
  stream.seekg(BMPWIDTH);
  stream.read(check, 2);
  width = bytesToInt(check, 2);

  // Grab the height of the image
  stream.seekg(BMPHEIGHT);
  stream.read(check, 2);
  height = bytesToInt(check, 2);

  stream.close();
  return width * height;
}

// Return pointer to the image data
char* FileReader::getImgData() {
  return imgBuffer;
}

// Return a pointer to the integers with all the goals that each bitmap should have
int* FileReader::getImgGoals() {
  std::ifstream stream(GOALFILENAME);
  char number;
  int n, lineCount = 0;

  // Grab the number of goals
  while (!stream.eof()) {
    stream >> n;
    lineCount++;
  }
  stream.close();

  // Allocate memory to store the goals
  int* imgGoals = static_cast<int*>(malloc(sizeof(int) * (lineCount + 1)));

  std::ifstream stream2(GOALFILENAME);
  *imgGoals = lineCount;
  lineCount = 1;
  while (!stream2.eof() && lineCount < *(imgGoals)) {
    stream2 >> *(imgGoals + lineCount);
    lineCount++;
  }
  stream2.close();
  return imgGoals;
}

// Function to convert bytes to an int
int FileReader::bytesToInt(char* bytes, int number) {
  int n;
  if (number == 4)
    n = static_cast<int>(*(bytes + 3) << 24 | *(bytes + 2) << 16 | *(bytes + 1) << 8 | *bytes);
  else if (number == 2)
    n = static_cast<int>(*(bytes + 1) << 8 | *bytes);
  else
    return -1;

  return n;
}

// Return the image size
int FileReader::returnSize() {
  return imageSize;
}
