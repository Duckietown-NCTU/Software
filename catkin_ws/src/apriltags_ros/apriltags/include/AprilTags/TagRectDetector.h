#ifndef TAGRECTDETECTOR_H
#define TAGRECTDETECTOR_H

#include <vector>

#include "opencv2/opencv.hpp"

#include "AprilTags/TagDetection.h"
#include "AprilTags/TagFamily.h"
#include "AprilTags/FloatImage.h"

namespace AprilTags {

class TagRectDetector {
public:
	
	const TagFamily thisTagFamily;

	//! Constructor
  // note: TagFamily is instantiated here from TagCodes
	TagRectDetector(const TagCodes& tagCodes) : thisTagFamily(tagCodes) {}
	
	std::vector<TagDetection> extractTags(const cv::Mat& image);
	
	std::vector<TagDetection> extractTags(const cv::Mat& image, std::vector<cv::Rect>& quad_proposals);
};

} // namespace

#endif
