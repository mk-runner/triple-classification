# triple classification based on link prediction

## 1.link prediction
Link prediction is a common knowledge representation application, which aims to predict a missing entity or relationship given a specific entity and relationship or two specific entities. More concretely, given <?, r, t>, predicted head entity h; given <h, ? ,t>, predicted relation r; known <h, r, ? >, predicted tail entity t. </br>
## 2.triple classification based on link prediction
The essence of triad classification is to judge whether each triple is correct or not. If an error occurs in a certain triple, at least one of h, r, and t has an error. Based on this fact, I proposed a new triple classification method based on link prediction, called TCLP, which first constructs three candidate triples for each triplet with link prediction, then calculates the number of repetitions between the original triplet and its candidate triples, and finally generates a series of binary labels through a fixed threshold.
### 2.1 construction of candidate triples for each triple
Let <h, r, t> be an original triple, three candidate triples are generated through link prediction. More specifically, given <?, r, t>, we demand to predict the head entity h, we therefore demand to construct a candidate set of the head entity H(r,t). With the candidate set H(r,t), we can calculate the score of each candidate head entity by link prediction, and assume that the candidate head entity h1 with the highest score is the final head entity. Accordingly, a new candidate triple <h1, r, t> is obtained. Similarly, given <h, ?, t> and <h, r, ?>, we can obtain two new candidate triples <h, r1, t> and <h, r, t1>, respectively.
### 2.2 Generation of binary labels
Assuming the absence of the head entity h, the relationship r, and the tail entity t, separately, three candidate triples (i.e., <h1, r, t>,  <h, r1, t>, and <h, r, t1>) can be constructed with link prediction. Next, this paper compares the original triples <h, r, t> with its three candidate triples pairwise, and records the number of repetitions. If the number of repetitions is greater than or equal to 2,  then this paper regards it as correct, otherwise, it is wrong.
