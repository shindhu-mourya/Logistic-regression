# Logistic-Regression(LR):
It is a classification technique. Geometric based, NB was probabilistic based. It is a classification technique. Geometric based, NB was probabilistic based. LR can be derived by geometric, probability or loss-function, but best is geometric. There are three ways to interperate.
If data is linearly separable in a plane by line. line is y =mx+c , in plan = W(transpose)x+b =0, x and W are vectors, b is a scaler, # # Assumption: - 
Classes are almost linearly separable. That is the best way to use LR. NB has conditional independence. K-NN is neighbor based. In LR, plane is divided in two parts, plus or minus points. Task is to find a plane, that will separate positive and negative points.

if Wtxi >0 Yi =+1 if Wtxi <0 Yi =-1 , LR is decision based. Optimal W = argMax(sum (YiWtXi)) --> Get Max value. LR can have many planes, we need best Wi, best plane. This is optimization problem. Wtxi is the distance from Xi to the plane. YiW(transpose)xi : +ve so plane is correctly classifies. if <0 means incorrectly classifies. Outlier can mess up the data. In LR we would want as many as positive values for positive place and as many as negative values for negative plane. This the optimization function of LR. Find out the optmimal W (max values of + and -)

Extreme outlier points can impact the model badly. max sum is out outlier prone. So our above fucntion has to fix to take care of outlier. Squashing is used for that purpose.

# Squashing in LR:
Instead of using signed distances directly, if signed distance is small, use as it, and if singed distance is large, make it a smaller value. If distance is huge, I wil not use huge value, I will much smaller. so we remove outlier. We use Sigmoid Function.

Try Plot(1/1+e^-x) to see the impact. X is the signed distance. sigma (0) = 0.5 Values are in between 0 to 1 instead of -infinity to infinity. So, it comes to probabilistic interpretation.

# Monotonic Function:
G(x) increases if x increases. if x1 > x2 then g(x1) > g(x2) then g(x) is set to be monotonic funtion. log(x) is defined when x>0 it is monotonic function. try plot (log(x)) on google.

# Optimization Equations:
W" = ArgMin( Sum (log(1+ E(-Yi W(t) Xi)) W = {w1,w2,w3,w4....wn} w is a feature. every feature I have weight associated. if W(t) X >0 then Y is positive else Y is negative.

# Weight Vector:
Weight vector is a D dim points.
W = {w1,w2,ws3....wn} we have D features. 
I have f1,f2,f3...fn , for every feature I have weight associated. 
Xq -> Yq
if W(t) X >0 then Y is positive else Y is negative.
If W(t) Xq >0 then Yq =1 else Y=-1
If weight of X increases, probability of Y also increases. 
# Regularization: Overfitting and Underfitting:
Zi = Yi Wi(t) Xi , W Transpose. if I pick W such that: (a) all traning points are correctly classified. (b) Zi -> infinity. Wi has to be + infinity or - infinity. We get best fit.

Overfitting - doing perfect job.

# Regularization
W* = ArgMin ( Sum ( Log (1+exp(-Yi W(t) Xi)))) + Lambda Wt * W we are minimizing both. 
Lambda part is regularization. first term is the loss term. 
# Sparsity:
W = w1,w,2,w3....wn
Solution to LR is set to be sparse  if many Ws are zero. 
if W vectror is sparse, solution of LR is also sparse.
L1 regularization creates sparsity since weight set to zero.
Elastic-Net
Use L1 norm and L2 norm W* = ArgMin ( Sum ( Log (1+exp(-Yi W(t) Xi)))) + Lambda Wt * W + lambda ||W square||

# Probabilistic Interpretation or derivation of Logistic Regression:
One way is geometry and simple algebra. another way is probability and third way is loss minimization.

# Naive Bayes:
(i) if features are real valued, we had gaussian dist. (ii) Class label is random var.

Read book  https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf
(iii) X and Y conditionalyy independent. 
Linear Regression = Gaussian Naive Bayes  + Bernouli

### Loss minimization interpretation of LR:

Remember  W* = ArgMin(Sum (Log (1+exp (-Yi W(t)Xi))))  from 1 to n
Zi = Yi W(t) Xi = Yi * F(Xi)
I want to minimize incorrectly classified points. that is the whole point of classification. 
I want to minimize my loss. +1 to incorrectly classified  and 0 for correctly classified.  so we want to minimize the loss that is       the loss function. 
  W* = argmin(sum(0_1 loss(Xi,Yi,w)))) 
  
 F(x) is differentiable if x is continious, in order to solve loss function. 
 
 ### HyperParameter Search/Optimization:
 Lambda =0, Overfitting
 Lambda =1 , Underfitting.
 Lambda in LR is a real value. Find right lambda.
 One way is grid search like brute force technique:
 take limited set of numbers of lambda
 search lambda in >= [0.0001,0.01, 0.1,1,10.....100...1000], some folks search lamda in very wide window. 
 Check cross validation error for each lambda, that should be minimal. 
# Column/Feature Standradization:
Matrix: Calculate mean and std dev of as matrix. this is Standardization. Even is logistic regression it is mandatory to perform column Standardization. if two features are on multiple scale, we need to standardize columns/features. It is also called mean centric or scale. So, all features are brought to the same scale.

# Feature Importance & Model Interpretability:
We have d features and optimal weight vector for each feature. so we get f1,f2,f3...fn features and weight w1,w2,w3...wn. assume we have all features independent as we know logistic regression from probabilistic stand point is - Gaussian naive base + Bernoulli distribution on class labels. If all features are independent then - we can get feature importance using Wj's. But in realistic it is not possible.

# Model Interpretability:
If I want to tell model is sensible or not, if Yq is +1 or -1, class label is positive or negative. I can pick up most important feature which has absolute weight value which is large. I can pick those features only. I can interpret based on weight if that person is male or female based on length of hair for example. if height is tall and hair length is short, I can say person is male.

# Colinearity or MultiColinearity:
If F(i) = A (Fj) + B so we can say Fi and Fj are colinear. If features have multicollinear dependency, we cannot use abs (weight) of feature. Implement dependencies and weight vector will also change.

Perturbation Technique: use for detecting multicollinearity.
in a matrix, add some small noise on cell value, and train your logistic regression again after adding some noise, if weights values are different that means features are colinear.

Nonlinear planes: Convert to linear before applying logistic regression.
The Most important aspects of ML/AI:
(i) Feature engineering. (ii) Bias - variance (iii) Data Analysis & Visualization.

# Performance Measurement of Models:
Accuracy = # Correctly classified points/Total number of points in D test

# Issues with accuracy:
(i) Case of imbalanced data.

# Confusion Matrix:
Does not take probability scores. Binary Classification task:two class (0,1)

TPR = TP/P TNR = TN/N FPR = FT/N FNR = FN/P

# Precesion: 
TP/TP+FP Recall = TP/P f1 sCORE= 2* (Pre*Recall)/(Pre+Recall)
