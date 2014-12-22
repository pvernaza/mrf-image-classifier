/**
   @mainpage 

   @section Executables

   @subsection binaryTrainBatch
 **/

#include "cv.h"

#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "classifier/BinarySubmodularImageClassifier.hh"
#include "util/Dvec.hh"
#include "display/LabelingViewer.hh"

/*
void getImages(DIR* imageDir, 
	       const string dirpath,
	       vector<std::pair<std::string, IplImage*> >& images);
*/

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cerr << "usage: <input dir> <input weights>" << endl;
    exit(1);
  }

  srand48(time(NULL));

  DIR* dir = opendir(argv[1]);
  string dirName(argv[1]);
  string weightsFile(argv[2]);
  //  string outputdir = string(argv[2]);
  
  assert(dir != NULL);

  Dvec *wvec0;

  cout << "Loaded file " << weightsFile << endl;
  ifstream file(weightsFile);
  wvec0 = DvecUtils::deserialize(file);
  file.close();
  
  BinarySubmodularImageClassifier imclass(wvec0, 1e6, 1e-5);

  cvNamedWindow("original",0);

  struct dirent *dirHandle = NULL;

  do {

    dirHandle = readdir(dir);

    string fileName = dirHandle->d_name;

    // skip non-png's
    if (fileName.find(".png") == string::npos) {
      continue;
    }

    string imPath = dirName + "/" + fileName;
    IplImage *inpImage = cvLoadImage(imPath.c_str());

    std::cout << "Loaded " << imPath << std::endl;

    cvShowImage("Original", inpImage);

    // segment the image
    IplImage* outSeg = 
      cvCreateImage(cvGetSize(inpImage), IPL_DEPTH_32S, 3);
    imclass.evaluate(inpImage, outSeg);

    // view and save the segmentation
    IplImage* dispSeg =
      cvCreateImage(cvGetSize(inpImage), 8, 3);
    LabelingViewer::viewLabeling("Segmented", 2, inpImage, outSeg, dispSeg);

    std::string outPath = 
      imPath.substr(0, imPath.find(".png")) + ".seg.png";
    cvSaveImage(outPath.c_str(), dispSeg);

    //    cvWaitKey();
    cvReleaseImage(&outSeg);
    cvReleaseImage(&dispSeg);
    cvReleaseImage(&inpImage);

  } while (dirHandle != NULL);
}

// caller must free images
// FIXME: adapt to other types of images
/*
void getImages(DIR* imageDir, 
	       const string dirpath,
	       vector<std::pair<std::string, IplImage*> >& images) {

  struct dirent *dent = readdir(imageDir);

  if (dent == NULL) 
    return;

  // skip non-png's
  string fname = dent->d_name;
  string::size_type dotpos = fname.find(".png");
  
  if (dotpos != string::npos) {
    string imPath = dirpath + "/" + fname;
    IplImage *image = cvLoadImage(imPath.c_str());
    images.push_back
      (std::pair<std::string, IplImage*>(imPath, image)); 
    cout << "L " << imPath << endl;
  }

  getImages(imageDir, dirpath, images);
}
*/
