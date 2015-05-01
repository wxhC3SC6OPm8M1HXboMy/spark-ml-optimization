# spark-ml-optimization

These are spark implementations for large-scale optimization for ML classifiers. There algorithms are supported:
IPA (iterative parameter averaging), PH (progressive hedging), and ADMM (alternating directions method of multipliers). 
The description of algorithms and implementations can be found on the blogs: http://dynresmanagement.com/blog.html (personal blog) or http://www.opexanalytics.com/blog

The optimization routines are in package optimization. 
The algorithms form the standard spark optimization data classes and thus can be used in the same way. The only difference is that the user can specify the learning rate function (in addition to specifying the step size factor, we also allow a custom step size function). Thus the learning rate fuction is: stepSize * stepSizeFunction(iteration_count). 
The reason for this is the fact that the default implementation in spark of the type stepSize/sqrt(iteration count) doesn't work well in general. 

For ADMM and PH there is also customization for solving the regularization problem. The class is: RegularizationOptimization. We wrote the L2 implementation (it would be easy to create a similar class for L1). 

Package classification creates logistic regression and SVM classes based on IPA, ADMM, PH. 

To test the method, one can specify the various parameters in the resource file src/resources/reference.conf (application.conf holds the default values of parameters). When running spark in the stand alone mode, you have to specify --conf "path/reference.conf"
Then use class Experiments to perform and test the code. 
