#include <stdlib.h>
#include <fstream>
#include <sstream>

// BMP file header offsets
#define DATAOFFSET 0xA  // 10 dec
#define DATASIZEINBYTES 0x22  // 34 dec
#define BMPWIDTH 0x12
#define BMPHEIGHT 0x16
#define BMPBPP 0x1C
#define COMP 0x1E

#define GOALFILENAME "goals.txt"

// Class to read in monochrome .bmp files
class FileReader {
 private:
  char* imgBuffer;  // Buffer to store image data
  char* check;
  bool firstImageRead;
  std::ifstream fs;
  int width;
  int height;
  int imageSize;
 public:
  FileReader();
  ~FileReader();
  // Function to read in the .bmp image
  bool readBitmap(int fileNum);
  // Function to grab the dimensions of the images
  int getBitmapDimensions();
  // Function to grab the target list
  int* getImgGoals();
  // Function to point to the currently read data
  char* getImgData();
  // Function to convert bytes into an int
  int bytesToInt(char* bytes, int number);
  // Function to find the endianess of the machine
  int returnSize();
};
