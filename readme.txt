Kirtan Amitkumar Jha
920392093

Q1 I am using the following parameters as a grading policy. 
1. manhattan distance between food and pacman
the minimum distance between pacman and food will be used for scoring as this distance is inversely proportional to better chance of winning

2. manhattan distance pacman and ghost
the minimum distance between pacman and ghost will be used for scoring as this distance is directly proportional to better chance of winning
I am using average of ghost distances

3. If distance between ghost and pacman is less than 2 units than pacman is very unlikely to win so this possibility must be assigned very low priority

Q2
Q3
Q4
Q5 I tried to use the same algo of question 1 by changing the state to current from successor but could not pass through the autograder.