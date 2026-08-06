# RZERO_8B — question samples


_R-Zero trained-questioner training set, Qwen3-8B (post-filter p̂∈[0.3,0.8])_

Per iteration: total pool size N, then an evenly-spaced sample of 40 questions.
Fields — `q`: question text; `label`: pseudo-label/answer used for training; `gt`: challenger-proposed answer (DEO only); `phat`: self-consistency p̂ = modal-count/m (R-Zero `score` is the same quantity); `runc`: uncertainty reward 1−2|p̂−0.5| (DEO only).


## iter 1  (N=3946, showing 40)

### 1. (p̂=0.375, r_unc=-)
**Q:** A rectangle with a perimeter of 30 units is divided into 3 equal smaller rectangles. What is the area of the original rectangle? Each of the smaller rectangles has a width that is half of the length of the original rectangle. Additionally, the total area of all three smaller rectangles combined is 75 square units. Calculate the dimensions of the original rectangle.
**label:** `50`

### 2. (p̂=0.333, r_unc=-)
**Q:** How many different combinations of three positive integers can you find to represent the sides of a right triangle if each integer is less than 20?
**label:** `5`

### 3. (p̂=0.667, r_unc=-)
**Q:** In a magical forest, there are $n$ trees arranged in a straight line, each with a height of $1, 2, 3, \ldots, n$ meters, respectively. A wizard casts a spell that causes each tree to grow an additional height equal to the sum of the heights of all the trees to its right. If the total new height of all trees is $1540$ meters, what is the original height of the $10$th tree from the left?
**label:** `10`

### 4. (p̂=0.778, r_unc=-)
**Q:** How many ordered triples \((x, y, z)\) of positive integers satisfy the equations \(x + y + z = 15\) and \(xyz = 105\)?
**label:** `6`

### 5. (p̂=0.556, r_unc=-)
**Q:** A number is called "double factorial" if it is the product of all the integers from 1 to n that have the same parity as n. For example, 6!! = 2 * 4 * 6. Find the smallest positive integer n such that n!! is a perfect square.
**label:** `1`

### 6. (p̂=0.556, r_unc=-)
**Q:** Let $x,$ $y,$ and $z$ be nonnegative real numbers such that $x + y + z = 1.$  Find the minimum value of
\[\frac{x^3 + y^3 + z^3}{x^2 + y^2 + z^2}.\]
**label:** `\frac{1}{3}`

### 7. (p̂=0.556, r_unc=-)
**Q:** In a classroom with 30 students, each student has chosen a favorite subject from a list of mathematics, science, literature, and art. If 10 students prefer mathematics, and the rest are divided such that twice as many prefer science as literature, and the number of students who prefer art is equal to the number who prefer literature, how many students prefer each of the subjects science, literature, and art? What is the percentage of students who prefer science?
**label:** `33.33\%`

### 8. (p̂=0.556, r_unc=-)
**Q:** Find all positive integers \( n \) for which \( n^4 + 4^n \) is a prime number.
**label:** `n = 1`

### 9. (p̂=0.333, r_unc=-)
**Q:** In a plane, there are 100 points, no three of which are collinear. How many triangles can be formed by connecting these points such that the area of each triangle is less than or equal to half the area of the triangle formed by connecting the three farthest points?
**label:** `80850`

### 10. (p̂=0.375, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(0) = 1 \) and \( P(1) = 2023 \). Find the maximum possible number of integer roots of \( P(x) \).
**label:** `0`

### 11. (p̂=0.667, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that \( n^2 \equiv 1 \pmod{7} \) and \( n^3 \equiv 1 \pmod{11} \).
**label:** `1`

### 12. (p̂=0.778, r_unc=-)
**Q:** What is the greatest integer \( n \) for which \( n^2 \le 100 \)?
**label:** `10`

### 13. (p̂=0.333, r_unc=-)
**Q:** A regular hexagon has a perimeter of 12 units. What is the sum of the squares of the distances from any one vertex to each of the remaining vertices?
**label:** `48`

### 14. (p̂=0.375, r_unc=-)
**Q:** In a convex quadrilateral \(ABCD\), the diagonals \(AC\) and \(BD\) intersect at point \(P\). Given that \(AP = 3\), \(PC = 6\), \(BP = 4\), and \(PD = 9\), find the ratio of the area of triangle \(APB\) to the area of triangle \(CPD\). Express your answer as a common fraction.
**label:** `\frac{1}{2}`

### 15. (p̂=0.375, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(1) = P(3) = P(5) = P(7) = a \) and \( P(2) = P(4) = P(6) = P(8) = b \), where \( a \neq b \). Find the minimum possible value of \( a - b \).
**label:** `15`

### 16. (p̂=0.333, r_unc=-)
**Q:** Pentagon $ABCDE$ is circumscribed about a circle. If $AB = 11$, $BC = 10$, $CD = 13$, $DE = 12$, and $AE = x$, what is the perimeter of pentagon $ABCDE$?
**label:** `51`

### 17. (p̂=0.778, r_unc=-)
**Q:** Alice is planning a treasure hunt in a square garden. She places a treasure at the corner of the garden and drops 5 other clues along the perimeter. Each clue points to the next clue, and each clue is positioned at a distance that forms an arithmetic sequence starting from the first clue. If the first clue is placed 2 meters from the treasure and the total distance around the garden covered by all clues is 30 meters, how far apart are the third and fifth clues?
**label:** `4`

### 18. (p̂=0.444, r_unc=-)
**Q:** In a far-off kingdom, there exists a magical tree with exactly 2023 branches. Each branch splits exactly once, forming two smaller branches, until it reaches the end, at which point it terminates in a leaf. A wizard casts a spell that causes every 13th leaf to glow. If the wizard casts this spell a total of 10 times, how many glowing leaves will there be in the tree after the last spell has been cast?
**label:** `1550`

### 19. (p̂=0.667, r_unc=-)
**Q:** There are 8 people in a room. Each person knows exactly 4 others in the room. If two people know each other, they are considered friends. What is the minimum number of friends any person can have?
**label:** `4`

### 20. (p̂=0.444, r_unc=-)
**Q:** Define a 'good' triangle to be a triangle in the xy-plane with all sides parallel or perpendicular to the axes.  An isosceles triangle with a fixed orientation of one side, say $\overline{AB}$, and of given length, say $AB = 2a$, is inscribed into a unit circle centered at the origin.  How many distinct 'good' triangles, that is the vertices of which lie on the circle, can be inscribed in the triangle $ABC$?

Given that the coordinates of point $A$ are $(a,0)$ and of point $B$ are $(-a,0)$, find the number of different 'good' triangles that can be inscribed in the triangle $ABC$.
**label:** `4`

### 21. (p̂=0.571, r_unc=-)
**Q:** Determine the minimum number of coins required to pay a debt of $20.16 using quarters ($0.25), dimes ($0.10), nickels ($0.05), and pennies ($0.01). Each type of coin must be used at least once.
**label:** `83`

### 22. (p̂=0.778, r_unc=-)
**Q:** In a complex plane, a function $f(z)$ maps a circle centered at the origin with radius $r$ to another circle centered at the origin with radius $2r$. If $z = x + yi$ is a complex number on the original circle, and $f(z) = z^2 + z$, find the value of $r$ that ensures $f(z)$ remains on the new circle for all $z$.
**label:** `1`

### 23. (p̂=0.333, r_unc=-)
**Q:** Let $P(x)$ be a polynomial with integer coefficients such that $P(1) = 5$ and $P(2) = 10$. If $P(n)$ is divisible by 7 for some positive integer $n > 2$, what is the minimum possible value of $n$?
**label:** `5`

### 24. (p̂=0.556, r_unc=-)
**Q:** A rectangular prism has three faces with areas 24, 48, and 72 square units. What is the volume of the prism?
**label:** `288`

### 25. (p̂=0.778, r_unc=-)
**Q:** Alice and Bob play a game with a deck of 52 cards, divided into 4 suits, each with 13 cards. They take turns drawing cards from the deck without replacement. The game ends when a player draws a card of the same suit as the card they drew in their previous turn. If Alice starts first, what is the probability that she draws the last card of the game?
**label:** `\dfrac{1}{2}`

### 26. (p̂=0.778, r_unc=-)
**Q:** Let \(a\) and \(b\) be real numbers. If the line represented by the equation \(ax + by = 4\) passes through the midpoint of the line segment with endpoints \((1, -3)\) and \((-1, 1)\), and is perpendicular to the line \(2x + 3y = 6\), what is the ordered pair \((a, b)\)?
**label:** `(6, -4)`

### 27. (p̂=0.444, r_unc=-)
**Q:** An arithmetic sequence starts with a first term of 3 and has a common difference of 2. After the 10th term, the sequence becomes a geometric sequence with the common ratio of 2. Find the sum of the first 15 terms of this combined sequence.
**label:** `771`

### 28. (p̂=0.667, r_unc=-)
**Q:** A regular octagon is inscribed in a circle of radius 10 units. Find the area of the octagon, expressed in simplest radical form.
**label:** `{200}\sqrt {2}`

### 29. (p̂=0.750, r_unc=-)
**Q:** A sequence of positive integers  $a_n$  is defined recursively by  $a_{n+1} = a_n^2 + 1$  for all  $n \geq 1$ . Determine the smallest positive integer  $k$  such that  $a_k > 10^{10}$ .
**label:** `7`

### 30. (p̂=0.333, r_unc=-)
**Q:** Let \( P(x) \) be a monic polynomial with integer coefficients such that \( P(0) = 1 \) and all roots of \( P(x) \) are integers. If \( P(x) \) can be factored into irreducible polynomials over the integers as \( P(x) = Q_1(x) Q_2(x) \cdots Q_n(x) \), where \( Q_i(x) \) are monic and have no repeated roots, find the minimum possible value of \( n \).
**label:** `1`

### 31. (p̂=0.429, r_unc=-)
**Q:** Let \( S \) be the set of all positive integers that can be expressed as the sum of the squares of two distinct positive integers. Determine the smallest positive integer \( n \) such that \( n \) is not in \( S \).
**label:** `1`

### 32. (p̂=0.667, r_unc=-)
**Q:** What is the minimum value of $a^2+b^2+c^2+d^2$ if real numbers $a,$ $b,$ $c,$ $d$ satisfy the equation $a+b+c+d=10$?
**label:** `25`

### 33. (p̂=0.556, r_unc=-)
**Q:** There are 12 identical-looking coins, among which there is exactly one counterfeit coin. Using an electronic balance, what is the minimum number of weighings needed to determine if the counterfeit coin is heavier or lighter than the other 11 coins?
**label:** `3`

### 34. (p̂=0.556, r_unc=-)
**Q:** Let \( P(x) = x^4 + ax^3 + bx^2 + cx + d \) be a polynomial with real coefficients such that \( P(1) = 17 \) and \( P(-1) = -17 \). Determine the number of distinct real roots of the equation \( P(x) = 0 \).
**label:** `2`

### 35. (p̂=0.556, r_unc=-)
**Q:** Consider a regular 2023-gon inscribed in the unit circle in the complex plane. Let \( P \) be the product of the distances from one vertex to all other vertices. Find the remainder when the integer part of \( P \) is divided by 1000.
**label:** `23`

### 36. (p̂=0.333, r_unc=-)
**Q:** Prove that the product of any three consecutive positive integers is divisible by 6.
**label:** `6`

### 37. (p̂=0.444, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(0) = 1 \) and \( P(1) = 3 \). Define a sequence \( \{a_n\} \) by \( a_1 = 1 \) and \( a_{n+1} = P(a_n) \) for \( n \geq 1 \). Prove that there exists a positive integer \( k \) such that \( a_k = 0 \).
**label:** `k`

### 38. (p̂=0.556, r_unc=-)
**Q:** Let $P(x)$ be a monic polynomial of degree 6 with real coefficients, and suppose that $P(x)$ has six distinct real roots. If the polynomial $Q(x) = x^6P(1/x)$ also has six distinct real roots, prove that the sum of these roots is an integer. Determine the value of this sum.
**label:** `0`

### 39. (p̂=0.375, r_unc=-)
**Q:** If \(a\) and \(b\) are real numbers satisfying \((\sum_{n=1}^{\infty} \frac{(-1)^n n}{n^2 + a}) + b = 0\), find the value of \(a\).
**label:** `1`

### 40. (p̂=0.333, r_unc=-)
**Q:** What is the area, in square units, of trapezoid $ABCD$ in the figure shown if points $A$, $B$, $C$, and $D$ are coplanar, angle $D$ is a right angle, $AB = 9$ units, $BC = 12$ units, $CD = 15$ units, and $AD = 10$ units?
**label:** `120`


## iter 2  (N=3977, showing 40)

### 1. (p̂=0.444, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(0) = 1 \) and \( P(2023) = 2023 \). Suppose that for every integer \( k \), there exists an integer \( n \) such that \( P(n) \equiv k \pmod{2023} \). Find the smallest possible degree of \( P(x) \).
**label:** `1`

### 2. (p̂=0.333, r_unc=-)
**Q:** Let $ABCD$ be a cyclic quadrilateral.  Let $P$ be the intersection of the diagonals of quadrilateral $ABCD.$  Let $E,$ $F,$ and $G$ be the feet of the perpendiculars from $P$ to the lines $AB,$ $BC,$ and $AD,$ respectively.  Let $T$ be the point on the circumcircle of triangle $PCG$ such that $\angle {PTG} = 90^{\circ}.$  What is the value of $PT \cdot ET$? [asy]
size(120);
pair A,B,C,D,E,F,G,H,I,J,K,L;
A=(0,0);
B=(4,0);
C=(4,3);
D=(0,3);
E=(2,1.5);
F=(1.5,1.5);
G=(0,1.5);
H=(2,0);
I=(3,3);
J=(3,0);
K=(2.5,2.5);
L=(1.5,0);
draw(A--B--C--cycle);
draw(D--E--F--cycle);
draw(A--D);
draw(B--C--A);
draw(C--D);
draw(E--F);
draw(G--H--I);
draw(H--J--G);
draw(K--L--J);
label("$A$",A,NW);
label("$B$",B,E);
label("$C$",C,NE);
label("$D$",D,W);
label("$E$",E,NE);
label("$F$",F,N);
label("$G$",G,W);
label("$P$",H,SW);
label("$T$",K,W);
label("$X$",L,W);
[/asy]
**label:** `1`

### 3. (p̂=0.778, r_unc=-)
**Q:** Alice and Bob play a game with a polynomial $f(x) = x^3 + ax^2 + bx + c$, where $a$, $b$, and $c$ are integers. Alice picks any polynomial $g(x)$ of degree at most 3 with integer coefficients. Bob then chooses an integer $x$, and the game ends. The winner is determined by whether the value $f(g(x))$ is divisible by 5 or not. If $f(g(x))$ is divisible by 5, Alice wins; otherwise, Bob wins. What is the smallest positive value of $f(1)$ that Alice can guarantee to win the game, regardless of Bob's choice of $x$ and Bob's subsequent choice of $g(x)$?
**label:** `5`

### 4. (p̂=0.333, r_unc=-)
**Q:** Given a sequence of positive integers \(a_1, a_2, a_3, \ldots\), let \(S_n\) denote the sum of the first \(n\) terms of the sequence. Suppose that for every positive integer \(n\), the sequence \(S_n, S_{n+1}, S_{n+2}\) forms an arithmetic progression. Determine the least positive integer \(k\) for which it is possible that \(a_k = 100\).
**label:** `2`

### 5. (p̂=0.333, r_unc=-)
**Q:** A sequence \((a_n)\) is defined by \(a_1 = 1\) and \(a_{n+1} = \frac{a_n + 2023}{a_n + 1}\) for all \(n \geq 1\). Find the smallest positive integer \(k\) such that \(a_k\) is an integer.
**label:** `3`

### 6. (p̂=0.667, r_unc=-)
**Q:** Mary has a garden of 120 plots, each of which can accommodate either carrots or beets. She decides to plant a certain number of plots with carrots and the rest with beets. The ratio of carrot seeds to beet seeds varies such that if she plants an integer \( x \) plots with carrots, the number of seeds required for carrots is given by the polynomial \( P(x) = 3x^2 - 10x + 4 \) and for beets by \( Q(x) = -2x^2 + 24x + 15 \). Mary wants to minimize the total number of seeds used in her garden while still planting both vegetables. Determine the number of plots she should plant with carrots to achieve this.
**label:** `0`

### 7. (p̂=0.333, r_unc=-)
**Q:** In triangle \(ABC\), \(\angle ABC = 90^\circ\) and \(\angle BAC = 60^\circ\). Let \(M\) be the midpoint of \(AC\), and let \(D\) be a point on \(BC\) such that \(MD\) is perpendicular to \(AC\). If \(AB = 6\) and the area of triangle \(ABM\) is \(9\sqrt{3}\), find the length of segment \(BD\).
**label:** `3\sqrt{3}`

### 8. (p̂=0.556, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial of degree 4 with integer coefficients. If \( P(1) = 17 \), \( P(2) = 34 \), and \( P(3) = 51 \), find the sum of all possible values of \( P(4) \).
**label:** `68`

### 9. (p̂=0.667, r_unc=-)
**Q:** A mathematician writes a sequence of numbers $a_1$, $a_2$, $\ldots$, $a_n$ on the board. For each $i$ (1 ≤ $i$ < $n$), he also writes the value $\frac{a_i}{a_{i+1}}$ next to it. After all numbers are written down, the mathematician erases all numbers except for the ones adjacent to every $\frac{a_i}{a_{i+1}}$ value. This process leaves exactly $n-1$ distinct fractions on the board. Given that $n$ is a prime number and $a_i$ > 0 for all $i$, find the sum of all possible values of $n$ for which this setup is possible.
**label:** `2`

### 10. (p̂=0.667, r_unc=-)
**Q:** A sequence of positive integers \(a_1, a_2, a_3, \ldots\) is defined as follows: \(a_1 = 1\) and for \(n \geq 1\),
\[
a_{n+1} = 
\begin{cases} 
\sqrt{a_n} & \text{if } \sqrt{a_n} \text{ is an integer}, \\
2a_n & \text{otherwise}.
\end{cases}
\]
Determine the smallest integer \(k > 1\) such that \(a_k = a_1\).
**label:** `2`

### 11. (p̂=0.444, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(0) = 1 \) and \( P(1) = 2023 \). Suppose there exists a positive integer \( n \) such that \( P(n) = 2024 \) and \( P(n+1) = 2025 \). Determine the minimum possible degree of \( P(x) \).
**label:** `3`

### 12. (p̂=0.444, r_unc=-)
**Q:** Albert and Belinda are trying to decide whether or not they should watch a movie on a new streaming service. They assign the following utility scores for watching a movie:

- Albert: A movie gets +3 points for a funny ending, -1 points for a boring opening, and -3 points for predictable plot.
- Belinda: A movie gets +2 points for a funny ending, +1 point for a romantic subplot, and -4 points for predictable plot.

If they both like a movie, they get the sum of their scores; if they don't like it, they only get the max of their scores. Movies this month score as follows:

- "Parody Palooza": funny ending, boring opening, predictable plot
- "Heartstrings": boring opening, romantic subplot
- "Matrix Redux": funny ending, predictable plot

Determine the total number of movies that Albert and Belinda both like, that they both don't like, and that either of them likes but not both.
**label:** `3`

### 13. (p̂=0.556, r_unc=-)
**Q:** Alice and Bob are playing a game with a fair coin and a standard 6-sided die. Alice flips the coin. If it lands heads, she rolls the die once; if it lands tails, she rolls the die twice. The sum of the die rolls (if any) determines their score. If Alice's total score exceeds 10, Bob must guess whether Alice originally got heads or tails. If Bob guesses correctly, Alice wins; otherwise, Bob wins. If Alice's score does not exceed 10, no guessing is required, and Alice wins. What is the probability that Alice wins the game?
**label:** `\frac{47}{48}`

### 14. (p̂=0.444, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients, such that \( P(0) = 1 \) and \( P(1) = 2023 \). Suppose further that for some positive integer \( n \), \( P(x) \) has a root modulo \( n \). Determine the smallest possible value of \( n \) for which such a polynomial exists.
**label:** `7`

### 15. (p̂=0.333, r_unc=-)
**Q:** You have a 30x30 checkerboard. Each square is painted either black or white, such that no two adjacent squares share the same color. A 'strip' is defined as a connected region of the same color squares that are either horizontally or vertically adjacent (not diagonally). How many distinct 'strip' sizes are possible, and what is the sum of all possible 'strip' sizes?
**label:** `465`

### 16. (p̂=0.556, r_unc=-)
**Q:** Define a "good sequence" as a non-empty set of consecutive integers whose sum is divisible by 2024. What is the maximum length of a good sequence chosen from the set {1, 2, 3, ..., 2023, 2024}?
**label:** `2024`

### 17. (p̂=0.333, r_unc=-)
**Q:** Let $ABC$ be a right triangle with $\angle ACB = 90^\circ$ and $AC = 3600$. A circle $\omega$ is inscribed in $ABC$ and is tangent to side $AB$ at point $X$. Let $P$ be the midpoint of the minor arc $AB$ of $\omega$ that does not contain $C$. If the distance from $P$ to line $AB$ is $240\sqrt{3}$, find the radius of $\omega$.
**label:** `240\sqrt{3}`

### 18. (p̂=0.333, r_unc=-)
**Q:** Consider a 2024 x 2024 grid of squares. Alice and Bob take turns coloring squares with Alice going first. On each turn, a player must color a square the same color as an adjacent square (sharing an edge) and may only color a square if it is not already colored. The game ends when no more moves are possible. Let \( N \) be the maximum possible number of squares that can be colored by the two players. Determine the remainder when \( N \) is divided by 1000.
**label:** `288`

### 19. (p̂=0.778, r_unc=-)
**Q:** Consider a \( 100 \times 100 \) grid of unit squares. A **magic path** is a sequence of moves from one square to another, such that each move is either to the right, upwards, left, or downwards, and no square is visited more than once. If a magic path starts from the bottom-left corner square and ends at the top-right corner square, what is the maximum possible number of squares that can be visited by such a path?
**label:** `10000`

### 20. (p̂=0.333, r_unc=-)
**Q:** Masha and Slava are taking turns selecting integers in ascending order from 1 to 50,000 inclusive (starting with Masha). Each number is picked exactly once. The loser is the one who gets stuck at the number 1. Let’s introduce a strict turn penalty: if player X has the last move, they lose 1 point; if the last move is made by player Y, they lose 2 points. They each start with an initial score of 0 points. What is the highest possible score difference they can end with?
**label:** `2`

### 21. (p̂=0.750, r_unc=-)
**Q:** There are 10 points in a plane such that no three are collinear and none of the triangles formed by these points have area greater than 10 square units. What is the minimum number of non-overlapping triangles that can be formed with these points?
**label:** `8`

### 22. (p̂=0.778, r_unc=-)
**Q:** Consider a square grid of size n × n, where n is a positive integer. We want to place n identical non-attacking rooks on the grid such that no two rooks share the same row or column. Let R(n) denote the number of ways to place the rooks. Prove that for all n ≥ 1, R(n) is divisible by n and find a closed form expression for R(n) in terms of n.
**label:** `n!`

### 23. (p̂=0.444, r_unc=-)
**Q:** Determine all positive integers <span>λ</span> for which there exists an infinite sequence of distinct real numbers <span>{x_i}</span> such that <span>x_{n^2 + λ} = x_n^2 + λx_n</span> holds for all integers <span>n ≥ 0</span>.
**label:** `1`

### 24. (p̂=0.333, r_unc=-)
**Q:** Consider a triangle with vertices at coordinates $(x_1, y_1)$, $(x_2, y_2)$, and $(x_3, y_3)$ in a plane. It is known that $x_1 + x_2 + x_3 = 30$ and $y_1 + y_2 + y_3 = 45$. Find the sum of the squares of the areas of the triangles formed by the midpoints of the sides of the original triangle.
**label:** `\frac{1}{16} A^2`

### 25. (p̂=0.444, r_unc=-)
**Q:** For all \(x\) and \(y\) real such that \[x^{3}-3xy^{2}=69\quad\text{and}\quad y^{3}-3x^{2}y=-17\], compute \(x^{2}+y^{2}\).
**label:** `8`

### 26. (p̂=0.556, r_unc=-)
**Q:** Prove that for all integers \( n \geq 1 \) and real numbers \( a \) and \( b \),
\[
\left\lfloor \frac{\left\lfloor na \right\rfloor + \left\lfloor nb \right\rfloor}{n} \right\rfloor \geq \left\lfloor a \right\rfloor + \left\lfloor b \right\rfloor.
\]
**label:** `\left\lfloor \frac{\left\lfloor na \right\rfloor + \left\lfloor nb \right\rfloor}{n} \right\rfloor \geq \left\lfloor a \right\rfloor + \left\lfloor b \right\rfloor`

### 27. (p̂=0.333, r_unc=-)
**Q:** Consider the polynomial \( P(x) = x^3 + ax^2 + bx + c \), where \( a, b, \) and \( c \) are real numbers. Suppose that \( P(x) \) has three distinct real roots, \( r_1, r_2, \) and \( r_3 \), satisfying the following conditions:
1. \( r_1 + r_2 + r_3 = 0 \)
2. \( r_1r_2 + r_2r_3 + r_3r_1 = -3 \)
3. \( r_1r_2r_3 = 2 \)
Define the sequence \( \{x_n\} \) by \( x_1 = r_1 \) and \( x_{n+1} = P(x_n) \) for \( n \geq 1 \). Determine the smallest positive integer \( k \) such that \( x_k = r_1 \) and \( x_{k-1} \neq r_1 \).
**label:** `3`

### 28. (p̂=0.778, r_unc=-)
**Q:** What is the remainder when 5 is divided by 9?
**label:** `5`

### 29. (p̂=0.444, r_unc=-)
**Q:** Let \(P(x)\) be a polynomial with integer coefficients such that \(P(0) = 1\) and \(P(1) = 2023\). Suppose further that there exists a positive integer \(n\) such that \(P(n) = n^{2023}\). Find the smallest possible value of \(n\) that satisfies these conditions.
**label:** `2023`

### 30. (p̂=0.444, r_unc=-)
**Q:** Let $N$ be a 7-digit positive integer, consisting of digits 1, 2, 3, ..., 9, in some order. Let $M$ be the integer obtained by reversing the digits of $N$. If $N$ is divisible by $M$, what is the smallest possible value of $N/M$?
**label:** `1`

### 31. (p̂=0.444, r_unc=-)
**Q:** At the annual MathFest conference, participants are seated in a circular arrangement. Each participant is assigned a number from 1 to 1000, and they can only sit next to participants whose numbers are consecutive to their own. For example, participant 1 can sit next to participants 2 or 1000, participant 500 can sit next to participants 499 or 501, and so on. If the conference director wants to arrange the seating such that no three consecutive numbers (like 1, 2, 3) are seated next to each other in any order, what is the minimum number of participants that must be seated to ensure this condition is met?
**label:** `1000`

### 32. (p̂=0.667, r_unc=-)
**Q:** A rectangular prism has dimensions \( 10 \times 20 \times 30 \) and is inscribed in a sphere. What is the volume of this sphere? Express your answer in terms of \(\pi\).
**label:** `\frac{7000\sqrt{14}}{3} \pi`

### 33. (p̂=0.778, r_unc=-)
**Q:** If every integer \( n \geq 2 \) is assigned to one of the sets \( S_1, S_2, S_3, \ldots \) based on the smallest prime factor of \( n \), with each set corresponding to a unique smallest prime factor, prove or disprove that for every pair of distinct sets \( S_i \) and \( S_j \), there exists an integer \( k \) such that the sum of the digits of \( k \) in base 10 is a prime number and \( k \) belongs to both \( S_i \) and \( S_j \).
**label:** `True`

### 34. (p̂=0.333, r_unc=-)
**Q:** Given the set $S = \{1, 2, 3, \ldots, 100\}$, an alternating permutation is a sequence $a_1, a_2, \ldots, a_{100}$ where $a_i < a_{i+1}$ if $i$ is odd, and $a_i > a_{i+1}$ if $i$ is even. How many such permutations are there such that for every $i$, if $a_i$ is a prime number, then $a_{i+1}$ is not a prime number, and if $a_i$ is not a prime number, then $a_{i+1}$ is a prime number? (Consider $a_{101}$ to be 101.)
**label:** `25! \times 75!`

### 35. (p̂=0.333, r_unc=-)
**Q:** If \(a\) is a prime number greater than 3, and \(b\) is the smallest positive integer such that \(a^2 + b\) is divisible by \(a + 2\), what is the remainder when \(b\) is divided by 6?
**label:** `5`

### 36. (p̂=0.333, r_unc=-)
**Q:** A rectangular piece of paper is folded in half vertically, and then in half again horizontally. After unfolding, a small square is removed from one corner and the paper is folded along the creases so that the corner opposite to the cut is on top. Given that the folded paper measures 2 units by 4 units, and the cut removes a square with side length 1 unit, what is the total area of the paper that is not visible from the outside of the folded paper when viewed from the corner opposite to the cut?
**label:** `15`

### 37. (p̂=0.333, r_unc=-)
**Q:** Ned and Nate each think of an arbitrary positive integer; then, they add both numbers and obtain a new positive integer. Exactly 970,015 times, out of 2,000,000 trials, the sum can be written as 165 times another positive integer. How many possible values of the number Ned is thinking of are there?
**label:** `165`

### 38. (p̂=0.667, r_unc=-)
**Q:** Alice, Bob, and Carol are playing a game with numbered tiles. Initially, Alice has 20 tiles numbered from 1 to 20, Bob has 15 tiles numbered from 1 to 15, and Carol has 10 tiles numbered from 1 to 10. They take turns placing a tile on a table, and the first person to place all their tiles wins. However, they must follow the rule: If Alice places a tile with a number \( n \), then the next player must place a tile with a number that is either a divisor or a multiple of \( n \) (but not \( n \) itself). Assuming optimal play by all, and starting with Alice, what is the probability that Carol wins?
**label:** `0`

### 39. (p̂=0.778, r_unc=-)
**Q:** In the plane, consider a set of \( n \) points \( P_1, P_2, \ldots, P_n \) with the following properties: 

1. Every point \( P_i \) is at a distinct distance \( d_i \) from the origin, where \( d_i > 0 \) and \( d_i \neq d_j \) for all \( i \neq j \).
2. For every pair of points \( P_i \) and \( P_j \) (\( i \neq j \)), the line segment connecting \( P_i \) and \( P_j \) is either vertical or horizontal.

For a given positive integer \( n \), determine the smallest number of distinct distances \( k \) that ensures the existence of such a set of points \( P_1, P_2, \ldots, P_n \).
**label:** `n`

### 40. (p̂=0.444, r_unc=-)
**Q:** You have a  $2023 \times 2023$  board divided into  $2023^2$  unit squares. If  $k$  is a positive integer less than  $2023$ , then a  $k\times k$  block is a block consisting of  $k^2$  unit squares and form a  $k\times k$  square. Find the smallest number  $n$  with the property that you can select  $n$  unit squares from the  $2023 \times 2023$  board (you are allowed to repeat unit squares) such that for any unit square on the board there exists a  $k\times k$  block that contains it and shares exactly one unit square with the set  $\{ \text{the } n \text{ selected squares}\}$ .
**label:** `2023`


## iter 3  (N=4129, showing 40)

### 1. (p̂=0.333, r_unc=-)
**Q:** Consider the set \( S \) of all ordered triples \( (x, y, z) \) of non-negative real numbers that satisfy the equation \( x^2 + y^2 + z^2 = 1 \). Let \( P \) be a point in \( S \) chosen uniformly at random. What is the probability that the distance from \( P \) to the origin is at least \(\frac{\sqrt{2}}{2}\)? Express your answer as a common fraction.
**label:** `1`

### 2. (p̂=0.333, r_unc=-)
**Q:** Define a *prime-step number* as a positive integer \( n \) whose digits can be rearranged such that the resulting number is a prime and all digits from 1 to 9 (at least once each) must appear in the digits of \( n \). For example, 123456789 is a prime-step number because it can be rearranged to form the prime number 235798146.

Determine the smallest integer \( k \) such that there exists a sequence of \( k \) distinct prime-step numbers \( n_1, n_2, \ldots, n_k \) where each pair of consecutive terms in the sequence forms a prime number when concatenated.

Note: The concatenation of two numbers \( a \) and \( b \), denoted \( ab \), is the number formed by appending the digits of \( b \) to the digits of \( a \).

For example, consider the numbers 23 and 35. When concatenated, 2335 is a 4-digit number.

Here, \( 2335 \) is a prime number.
**label:** `2`

### 3. (p̂=0.444, r_unc=-)
**Q:** What is the largest integer $N$ for which there exists a polynomial $P(x)$ with integer coefficients such that $P(1)$, $P(2)$, $\ldots$, $P(N)$ are perfect squares and $P(N+1)$ is not a perfect square?
**label:** `2`

### 4. (p̂=0.444, r_unc=-)
**Q:** In the town of Mathville, there are 100 houses arranged in a straight line. Each house is painted either red or blue. A house is considered "cool" if it is painted blue and the product of the number of blue houses to its left and the number of blue houses to its right is a perfect square. At the start of the year, exactly 50 houses are painted blue, and the rest are red. Throughout the year, houses change colors once per month in a sequence that ensures every possible arrangement of blue and red houses occurs exactly once. Determine the maximum number of "cool" houses Mathville can have at any single point during the year.
**label:** `50`

### 5. (p̂=0.556, r_unc=-)
**Q:** Consider a convex polygon \( P \) inscribed in a unit circle with \( n \) sides, where \( n \geq 4 \) is an integer. Each vertex of \( P \) is connected to every other vertex, forming a complete graph on \( n \) vertices. A “good” path in this graph is defined as a path where the number of vertices with a degree greater than 4 is minimized, and every vertex lies on exactly two good paths. Determine the smallest integer \( k \) such that for any convex polygon \( P \) and any two “good” paths within this polygon, there exists a point lying on both paths that lies on at least one side of \( P \).
**label:** `2`

### 6. (p̂=0.444, r_unc=-)
**Q:** Let \(P(x)\) be a polynomial with integer coefficients, and let \(Q(x)\) be a polynomial of degree 2 with integer coefficients. Define \(R(x) = P(x) + Q(x)\). Suppose that \(R(1) = R(2) = R(3) = R(4) = R(5)\), but \(R(6) \neq R(1)\). Determine the minimum possible value of \(|P(0)|\).
**label:** `1`

### 7. (p̂=0.333, r_unc=-)
**Q:** In a triangle \( ABC \), let \( \omega \) be a circle passing through \( B \) and \( C \) and intersecting \( AC \) and \( AB \) again at points \( P \) and \( Q \) respectively. Let \( M \) be the midpoint of segment \( PQ \). The circumcircle of \( \triangle BMQ \) intersects \( AC \) at \( N \neq C \). The line \( BN \) intersects \( \omega \) at \( D \neq B \). Given that \( DP = 10 \) and \( MN = 12 \), find \( AD \).
**label:** `20`

### 8. (p̂=0.444, r_unc=-)
**Q:** Three unit regular pentagons share a common center, with one vertex from each regular pentagon attached to the center. What is the angle, in degrees, between a line segment connecting the center to a vertex on one of the regular pentagons and a line segment connecting the center to a vertex on the adjacent regular pentagon that is not shared with the first one?
**label:** `72`

### 9. (p̂=0.444, r_unc=-)
**Q:** On an infinite, unbounded chessboard, every square $(x, y)$, where $x$ and $y$ are integers, is colored either black or white. Define a white connection as a pair of adjacent white squares $(x, y)$ and $(x \pm 1, y)$ or $(x, y \pm 1)$, which can be traversed in a sequence of at most $k$ steps to reach any other white square. For each positive integer $k$, find the smallest integer $N(k)$ such that for any such coloring, there exists a white connection with a distance at most $N(k)$ from $(0, 0)$.
**label:** `2k`

### 10. (p̂=0.778, r_unc=-)
**Q:** An infinite sequence of positive integers {a<sub>n</sub>} satisfies the following properties:
- a<sub>1</sub> = 2
- a<sub>n+1</sub> - a<sub>n</sub> is an integer power of 2 for all n ∈ ℕ.
- gcd(a<sub>n</sub>, a<sub>n+1</sub>) > 1 for all n ∈ ℕ.

For how many positive integers k, with 1 ≤ k ≤ 1000, can we guarantee that gcd(a<sub>k</sub>, a<sub>k+1</sub>) is divisible by 2<sup>k</sup>?
**label:** `1000`

### 11. (p̂=0.778, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(0) = 1 \) and \( P(1) = 2023 \). Suppose further that for every integer \( n \), \( P(n) \) is either a prime number or \( -1 \). Determine the maximum possible degree of \( P(x) \).
**label:** `1`

### 12. (p̂=0.556, r_unc=-)
**Q:** Among all quadruples  $(a, b, c, d)$  of nonnegative integers that sum to 1000, find the number that maximizes  $ad - bc$ .

(It may help to note that  $ad - bc = (a + b)(c + d) - (b + c)(a + d)$ , though this need not be in the answer.)
**label:** `250000`

### 13. (p̂=0.778, r_unc=-)
**Q:** Find all functions \( f: \mathbb{R} \rightarrow \mathbb{R} \) satisfying
\[ f(x^4 + f(y)) = y + f(x^2)^2 \]
for all \( x, y \in \mathbb{R} \).
**label:** `f(x) = x`

### 14. (p̂=0.444, r_unc=-)
**Q:** There are 1000 lamps labelled 1 through 1000, and 1000 corresponding buttons labelled 1 through 1000. The lamps are initially all on. When button labelled `i` is pressed, the state of lamp `i` (on or off) toggles, as well as the states of lamps `i+1` (if any) and `i-1` (if any). A sequence of button presses is performed as follows. An integer k is picked uniformly at random from {1,2,…,1000}. Then, for i=1,2,…,1000, button i is pressed with probability k / i. What is the probability that after this sequence, exactly one lamp is on?
**label:** `0`

### 15. (p̂=0.556, r_unc=-)
**Q:** Call a positive integer $N$ a 7-10 double if the digits of the base-$7$ representation of $N$ form a base-$10$ number that is twice $N$. For example, $51$ is a 7-10 double because its base-$7$ representation is $102$. What is the largest 7-10 double?
**label:** `315`

### 16. (p̂=0.556, r_unc=-)
**Q:** Alice and Bob play the following game on the Cartesian plane. Initially, Alice places a counter at the origin. Then, on each player’s turn, that player applies a rotation centered at the origin (clockwise or counterclockwise, of any desired angle) or a translation (by any desired distance, in any direction) to the counter, and keeps playing until the counter is at a point with integer coordinates. Alice goes first. What is the minimum number of moves after which Alice can guarantee that the counter is at a point with integer coordinates?
**label:** `1`

### 17. (p̂=0.333, r_unc=-)
**Q:** Consider a positive integer \( N \) with exactly 1000 divisors, each greater than 1. Define \( D(N) \) as the sum of the logarithms (base 10) of all divisors of \( N \). Let \( S \) be the smallest possible value of \( D(N) \). Determine the greatest integer less than or equal to \( S \).

It can be shown that the prime factorization of \( N \) can be expressed in the form \( p_1^{e_1} p_2^{e_2} \ldots p_k^{e_k} \) where \( e_i \geq 1 \) and \( e_1 \geq e_2 \geq \ldots \geq e_k \). Find the least possible value of \( k \).
**label:** `3`

### 18. (p̂=0.778, r_unc=-)
**Q:** In a game played on a board with 5 rows and 5 columns, there is a token on each of the 25 squares initially. Players take turns moving the token from one square to any adjacent (orthogonally or diagonally) empty square. After a total of 10 moves, the token will be on exactly 10 of the 25 squares. How many distinct sets of 10 squares could the token occupy, assuming it follows the rules above?
**label:** `3268760`

### 19. (p̂=0.444, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients satisfying \( P(0) = 1 \) and \( P(1) = 2 \). Furthermore, assume that \( P(x) \) can be expressed as a product of two non-constant polynomials \( Q(x) \) and \( R(x) \) with integer coefficients, where \( \deg(Q) + \deg(R) = 5 \). Prove that there exists a positive integer \( n \) such that \( P(n) \) is divisible by at least 10 distinct prime numbers.
**label:** `10`

### 20. (p̂=0.333, r_unc=-)
**Q:** At each vertex of a regular heptagon (7-sided polygon), a positive integer is placed such that the numbers at adjacent vertices always differ by at least 2. If the sum of all the numbers at the vertices equals 2014, what is the maximum possible number that could be placed at a single vertex?
**label:** `293`

### 21. (p̂=0.333, r_unc=-)
**Q:** Find all positive integers $n$ such that there exists a permutation $a_1, a_2, \ldots, a_n$ of $\{1, 2, \ldots, n\}$ satisfying the condition that the greatest common divisor (GCD) of the sums $a_1 + a_2 + \cdots + a_k$ and $n$ is 1 for each $k = 1, 2, \ldots, n$.
**label:** `1`

### 22. (p̂=0.778, r_unc=-)
**Q:** Alice and Bob play a game on a 5x5 grid of unit squares. Initially, Alice places one token on an empty square of the grid. Then, Bob places another token on a different square that is not directly adjacent to the square chosen by Alice (no shared side). They take turns placing tokens, each time avoiding squares that are adjacent (share a side) to any previously occupied square. The first person who cannot place a token loses. If Alice goes first, determine who has the winning strategy and why.
**label:** `Alice`

### 23. (p̂=0.667, r_unc=-)
**Q:** In the Cartesian plane, consider a sequence of points \( P_0, P_1, P_2, \ldots \) defined by \( P_0 = (0, 0) \) and for \( n \geq 0 \), \( P_{n+1} \) is the point obtained by rotating \( P_n \) counterclockwise by 60 degrees about the origin, and then translating the resulting point \( k \) units in the positive x-direction. Let \( Q_n \) be the midpoint of segment \( P_nP_{n+1} \). Find all positive integers \( k \) such that the area of the figure formed by the union of all triangles \( P_nQ_nP_{n+1} \) for \( n \geq 0 \) is finite.
**label:** `1`

### 24. (p̂=0.444, r_unc=-)
**Q:** Alice and Bob play a game on a 100x100 grid of unit squares. Alice chooses an initial non-empty set of squares. Bob then colors every square that shares a side with an odd number of colored squares. Alice wins if every square eventually becomes colored by Bob. Find the smallest initial set size such that Alice can guarantee a win.
**label:** `5000`

### 25. (p̂=0.444, r_unc=-)
**Q:** Let \( S(n) \) denote the number of positive integers less than \( 2^n \) that do not contain the sub-string '11' in their binary representation. Prove that for all positive integers \( n \geq 1 \), \( S(n) = n \cdot 2^{n-1} \).

Furthermore, consider the sequence \( T(n) = \sum_{k=1}^{n} (-1)^{k+1} S(k) \). Determine \( T(2024) \mod 1000 \).
**label:** `0`

### 26. (p̂=0.778, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(1) = 2 \) and \( P(2) = 3 \). Suppose \( P(x) \) has the property that for any prime number \( p \), \( P(p) \) is divisible by some prime number \( q \) not equal to \( p \). Find the smallest possible value of \( |P(0)| \).
**label:** `1`

### 27. (p̂=0.444, r_unc=-)
**Q:** In the plane, consider a regular hexagon with side length $1$. For a point $P$ inside the hexagon, define $f(P)$ as the minimum distance from $P$ to the six sides of the hexagon. Determine the largest constant $c$ such that $f(P) \leq c$ for all points $P$ inside the hexagon.
**label:** `\frac{\sqrt{3}}{3}`

### 28. (p̂=0.667, r_unc=-)
**Q:** Alice and Bob play a game on a 5 x 5 grid. Alice starts by placing a token on any square. Then, players take turns moving the token to an adjacent square (horizontally or vertically) that has not been visited before, with the restriction that a player cannot move to a square that would force the opponent into a losing position immediately. The game ends when no more moves are possible. The player who forces the game to end loses. If Alice always goes first, determine all starting positions from which she can force a win. What is the minimum number of squares she needs to occupy initially to guarantee victory, assuming both players play optimally?

Each square on the grid is numbered from 1 to 25 in a standard order. Describe your strategy and state the number(s) of the square(s) Alice should choose to win. How many different starting positions can Alice choose to guarantee a win? Find the smallest possible set of starting positions that will ensure Alice's victory. Can Alice find a set of less than 5 starting positions to force a win?

Answer the following questions based on Alice's strategy and the grid’s numbering:

1. What is the minimum number of squares Alice should occupy initially to guarantee victory?
2. List all the possible starting positions Alice can choose to guarantee a win.
3. Is it possible for Alice to guarantee a win with less than 5 starting positions? If yes, state the number(s) of the starting position(s).

Your response should include your mathematical reasoning and the answers to the questions above. You will be asked to justify your reasoning after the problem.
**label:** `13`

### 29. (p̂=0.444, r_unc=-)
**Q:** Define a divisor set for a positive integer $n$ as a set of pairwise coprime positive integers where no two elements differ by more than $3$ and at least one of the elements is coprime with $n$. What is the maximum possible number of elements in a divisor set of $n$?
**label:** `4`

### 30. (p̂=0.444, r_unc=-)
**Q:** Let \(P(x)\) be a polynomial with integer coefficients such that \(P(0) = 2024\). Suppose further that for any integer \(k\), \(P(k)\) divides \(P(P(k))\). Determine the largest possible value of \(P(-1)\).
**label:** `2023`

### 31. (p̂=0.444, r_unc=-)
**Q:** What is the least number of sticks needed to construct a fence around a rectangular courtyard such that each stick in the fence touches exactly 4 other sticks and the number of rows (a certain number of sticks arranged side by side in parallel rows) is equal to the number of columns (a certain number of sticks arranged end to end perpendicular to the rows)?
**label:** `4`

### 32. (p̂=0.333, r_unc=-)
**Q:** Alice and Bob play a game on a $2023 \times 2023$ grid. Initially, all cells are white. They take turns, starting with Alice, coloring an uncolored cell in black. Alice’s objective is to create as many $2 \times 2$ black squares as possible, while Bob’s is to minimize that number. What is the final number of $2 \times 2$ black squares on the grid, given both players use optimal strategies?
**label:** `0`

### 33. (p̂=0.556, r_unc=-)
**Q:** Prove that for any five positive real numbers \(a, b, c, d, e\), there exist positive real numbers \(x, y, z, w\) such that the polynomial equation \(x^3y + 5y^3z + 7z^3w + 2w^3x - axy^2 - byz^2 - czw^2 - dxw^2 - eyx^2 - fw^2x = 0\) holds. Furthermore, find the smallest positive integer \(N\) such that the polynomial can be expressed as a product of at most \(N\) polynomials each of degree at most 2.
**label:** `2`

### 34. (p̂=0.556, r_unc=-)
**Q:** Alice and Bob play a game on an infinitely large grid of unit squares. They take turns, starting with Alice. On each turn, a player must mark an unmarked unit square with their mark (Alice's mark is \( A \), and Bob's mark is \( B \)). The game ends when one of the players marks three squares in such a way that they form the vertices of an axis-aligned 1x2 rectangle. The first player to complete such a rectangle wins. If no player wins after all squares are marked, it's a draw. What is the least number of total marks Bob needs to guarantee a win?
**label:** `4`

### 35. (p̂=0.333, r_unc=-)
**Q:** We define a *prime-scan* sequence as a finite sequence of integers such that for any index \( i \), the product of the elements in the subsequence from the first element to the \( i \)-th element (inclusive) is a perfect square. Given that each element in the sequence must be a positive integer less than 100 and all elements are distinct, find the maximum possible length of a prime-scan sequence that includes at least one prime number.
**label:** `10`

### 36. (p̂=0.444, r_unc=-)
**Q:** Alice and Bob are playing a game on an infinite 2-dimensional grid. Initially, Alice places a token at the origin (0, 0). In each turn, Alice and Bob alternatively perform the following actions:

1. Alice's turn:
   - Alice selects an integer \(k\) where \(1 \leq k \leq 10\) and moves the token to a point \((x+k, y-k)\), \((x-k, y+k)\), \((x+k, y+k)\), or \((x-k, y-k)\) on the grid. The chosen direction is determined by a secret rule known only to Alice.
   
2. Bob's turn:
   - Bob, who does not know Alice's rule, tries to guess the direction by guessing a sequence of integers from 1 to 10 in any order. He submits a move in each of the four directions using the integers he guesses.

The game ends if Bob ever guesses the correct direction of Alice's move, in which case Bob wins. Otherwise, the game continues indefinitely.

Suppose that Alice knows the rules beforehand. What is the smallest number of integers Bob needs to guess in the worst-case scenario such that Bob is guaranteed to win the game?
**label:** `10`

### 37. (p̂=0.556, r_unc=-)
**Q:** Alice has a list of 100 positive integers, each of which is either 1 or a prime number. She wants to choose a subset of these numbers such that their product is a perfect square. What is the minimum number of elements Alice needs to remove from the list to ensure she can always find a subset whose product is a perfect square, regardless of the initial configuration of the list?
**label:** `50`

### 38. (p̂=0.778, r_unc=-)
**Q:** In the land of Numberia, every year the citizens celebrate the Festival of Digits. For the Festival, a special challenge is proposed: find a positive integer \(N\) such that when \(N\) is multiplied by any positive integer \(k\) (where \(1 \leq k \leq 9\)), the resulting product is a number that has the same digits as \(N\), just possibly rearranged. Note that a digit can be repeated.

This year, the Grand Council presents a twist: the integer \(N\) must be the smallest positive integer satisfying the conditions above, and its digit sum must not exceed 27. How many different values of \(N\) are there?
**label:** `1`

### 39. (p̂=0.333, r_unc=-)
**Q:** There are 2019 creatures on the monster carousel. Each creature has 3 eyes. Each eye can be one of four colors: red, blue, green, or yellow. Each creature also has a unique number of limbs ranging from 2 to 2020. 

Every carousel ride changes the eye color of exactly one creature's eyes while leaving their limbs unchanged. Two creatures are considered distinguishable if either their eye color patterns are different or they have different limb counts. 

After all possible combinations of eye colors have been exhausted, a creator joins the carousel with its initial eye color being white and zero limbs. After one full rotation around the carousel starting with this new creator, the eye colors and the limb count of each creature change following the previously described process. What is the minimum number of riders required to ensure that every possible configuration of creatures is generated?
**label:** `2019`

### 40. (p̂=0.667, r_unc=-)
**Q:** A 6-digit number of the form \(ABCDAB\) is formed using distinct non-zero digits \(A, B, C,\) and \(D\) with \(A < B < C < D\). This number is then tested for divisibility by 11, 13, and 31. It is found that none of these tests, without exception, generate numbers that satisfy the divisibility condition. What is the probability that \(A, B, C,\) and \(D\) can be chosen such that none of the numbers formed by their permutations, when entered in place of the digits of \(ABCDAB\), satisfy the divisibility test by 11, 13, or 31?
**label:** `1`


## iter 4  (N=4401, showing 40)

### 1. (p̂=0.333, r_unc=-)
**Q:** In a hypothetical network of cities, each city is represented by a unique integer from 1 to n. A magical train departs from city 1 at noon and visits each subsequent city in increasing order of integer label, adhering to the rule that it can only jump over exactly k cities from its current position. Upon arriving at a city, the train selects one resident at random for a special journey, with the probability of selection being proportional to the square of the number of letters in the city's name. After completing a tour that touches upon every city exactly once, what is the expected value of the sum of the numerical labels of the cities visited in descending order, given that n=25 and k=3?
**label:** `325`

### 2. (p̂=0.444, r_unc=-)
**Q:** Alice and Bob play a game on an infinite number line. They take turns performing moves, with Alice going first. On Alice's turn, she chooses an integer $n > 0$ and moves to the point $n$ units away from her current position. On Bob's turn, he chooses two distinct integers $a$ and $b$ and moves to the point $|a - b|$ units away from his current position. If after any move a player lands on a point that has already been visited by either player, they lose the game. Can Bob guarantee a win regardless of Alice's strategy? If so, what is the smallest positive integer $k$ for which Bob can guarantee a win after $k$ turns?
**label:** `1`

### 3. (p̂=0.333, r_unc=-)
**Q:** Alice has an \( n \times n \) grid, where each cell contains either a 0 or a 1. She defines a "valid path" from the top-left cell to the bottom-right cell as follows: the path must move only right or down, and at each step, the number in the next cell must be the same as the previous cell's number plus one (i.e., 1 goes to 2 and 0 goes to 1).

Alice starts from the top-left cell. She will perform a series of steps:
1. Pick any cell in the grid to mark it as "start."
2. From the "start" cell, follow any valid path until reaching either an invalid path or the bottom-right cell.
3. If the path reaches the bottom-right cell without any invalidations, mark the path as "completed."

Alice is curious: if she repeats this procedure multiple times with different "start" cells and valid paths, what is the minimum number of valid paths required to ensure that all possible "start" cells lead to at least one completed path, regardless of the initial configuration of the grid?

Find the minimum number of valid paths Alice must follow to guarantee this outcome for any \( n \times n \) grid.
**label:** `2`

### 4. (p̂=0.556, r_unc=-)
**Q:** In a universe governed by a peculiar set of rules, consider a square grid of size \(2024 \times 2024\) with each cell containing a unique integer from \(1\) to \(2024^2\). A "magic move" is defined as selecting any row or column and subtracting \(100\) from each of its cells, provided that after the operation, every cell contains a non-negative integer. Determine the minimum number of "magic moves" required to ensure that every cell in the grid contains a non-negative integer, under the constraint that no cell's value becomes negative.
**label:** `2024`

### 5. (p̂=0.333, r_unc=-)
**Q:** A rectangle is dissected into $5$ squares whose areas are $1, 9, 4, 4, 1$ respectively. What is the area of the colored hexagon? (Hint: Assemble the squares into a new rectangle in a creative manner.)
**label:** `18`

### 6. (p̂=0.333, r_unc=-)
**Q:** Alice and Bob play a game on an infinite grid of unit squares. Initially, all squares are white. Alice goes first, and they take turns coloring squares black. On each turn, a player can color any white square black, as long as it shares at least one side with a square that is already colored black. The game ends when no more moves are possible. Alice wins if the union of all colored squares forms a closed loop (i.e., there exists a continuous path along the boundaries of colored squares that starts and ends at the same square, going through no other colored square twice). Bob wins otherwise. Determine all integers \( n \geq 1 \) for which Alice has a winning strategy if the game starts with an \( n \times n \) board.
**label:** `n \geq 3`

### 7. (p̂=0.444, r_unc=-)
**Q:** Alice, Bob, and Carl play a game involving selecting non-empty subsets of the set \(\{1, 2, \ldots, 100\}\). The game proceeds in turns where Alice picks a non-empty subset \(A_0\), then Bob picks a non-empty subset \(B_0\), followed by Carl picking a non-empty subset \(C_0\). The game continues indefinitely with players alternating colors (Alice and Carl play white, Bob plays black) and selecting subsets based on the previous player's choice.

At the end of the game, two sums are computed:
- The white sum is the count of positive integers \(x\) that belong to \(A_0\), \(A_2\), \(A_4\), etc., but not to \(A_1\), \(A_3\), \(A_5\), etc.
- The black sum is the count of positive integers \(x\) that belong to \(B_1\), \(B_3\), \(B_5\), etc., but not to \(B_0\), \(B_2\), \(B_4\), etc.

The first player to make their sum equal to the other player's sum loses. Determine all possible sets \(A_0\) such that Alice can always win regardless of the strategies employed by Bob and Carl.
**label:** `\{1, 2, \ldots, 100\}`

### 8. (p̂=0.778, r_unc=-)
**Q:** Alice has a 10x10 grid of cells, initially all empty. She plays a game in which she can perform two types of moves: 
- **Type 1:** She can choose any row, and invert the parity of every cell in that row (i.e., if a cell was even, it becomes odd, and vice versa).
- **Type 2:** She can choose any column, and invert the parity of every cell in that column.

Alice is called **parity-proficient** if, after any series of moves, she can always set the parity of any single cell to even or odd, no matter what the current state of the grid is.

Alice is given a set of rules, each rule specifying a target parity for certain cells in the grid. She can win a prize if she can always achieve every rule's parity requirement. What is the largest number of rules such that Alice can win every rule?
**label:** `100`

### 9. (p̂=0.444, r_unc=-)
**Q:** Alice and Bob are playing a game on a rooted tree with 2024 nodes. The root of the tree is node 1, and each node has a unique integer label from 1 to 2024. The game proceeds in rounds. In each round, Alice selects a node x and sends it to Bob. Bob then selects a node y and sends it back to Alice. Alice wins if node x is an ancestor of node y. Otherwise, Bob wins.

Alice and Bob play optimally to maximize their chances of winning. If Alice plays first, what is the probability that Alice wins the game?
**label:** `\dfrac{1}{2}`

### 10. (p̂=0.333, r_unc=-)
**Q:** Consider a sequence of positive integers \( a_1, a_2, a_3, \ldots, a_n \) such that for each \( i \) from 1 to \( n-1 \), the difference \( a_{i+1} - a_i \) is either 1 or -1. Furthermore, let the sequence \( b_1, b_2, b_3, \ldots, b_n \) be defined by \( b_i = a_i^2 \) for each \( i \). If the sum \( b_1 + b_2 + \cdots + b_n = 2024 \) and the sequence has at least one increase (i.e., at least one \( a_{i+1} - a_i = 1 \)), what is the minimum possible value of \( n \)?
**label:** `44`

### 11. (p̂=0.333, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(0) = 1 \), \( P(1) = 2 \), and for every integer \( n \geq 2 \), \( P(n) \) divides \( P(n-1) + P(n+1) \). If \( Q(x) = P(x)P(x+1) \), find the smallest possible positive value of \( Q(2023) \).
**label:** `4098600`

### 12. (p̂=0.556, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(0) = 1 \) and \( P(1) = 2 \). Suppose further that for every prime number \( p \), the value \( P(p) \) is a multiple of \( p \). Define the sequence \( a_n \) by \( a_1 = 1 \) and \( a_{n+1} = P(a_n) \) for all \( n \geq 1 \). Determine the largest integer \( k \) such that \( a_k < 10^{10} \).
**label:** `9999999999`

### 13. (p̂=0.556, r_unc=-)
**Q:** Let $P(x)$ be a polynomial with integer coefficients such that $P(0) = 1$ and $P(1) = 3$. Suppose that for all integers $x$, if $P(x)$ is a perfect square, then $P(x+2)$ is also a perfect square. Determine the maximum possible number of integer roots of $P(x)$.
**label:** `0`

### 14. (p̂=0.333, r_unc=-)
**Q:** Let \( P(x) \) be a monic polynomial with integer coefficients such that \( P(1) = 2023 \). Suppose further that for every positive integer \( n \), there exists a positive integer \( m \) such that \( P(m) = n^3 + n^2 + n + 1 \). Determine the smallest possible value of \( P(0) \).
**label:** `2020`

### 15. (p̂=0.333, r_unc=-)
**Q:** Alice and Bob play a game on a rectangular grid of size \( n \times m \). Each cell contains a positive integer. Alice starts by placing a token on the top-left cell (1, 1). On her turn, a player can either move the token one cell to the right or one cell down, provided the move does not leave the grid. The game ends when the token reaches a cell containing a negative integer, or when all cells have been visited. If Alice moves first, what is the minimum number of cells she can guarantee to mark, regardless of Bob's moves? What is the minimum number of cells she can guarantee to mark if the game starts with Bob moving the token?
**label:** `\left\lceil \frac{n \times m}{2} \right\rceil`

### 16. (p̂=0.778, r_unc=-)
**Q:** Consider a sequence of integers \(a_1, a_2, a_3, \ldots\) defined by the recurrence relation \(a_{n+2} = 2a_{n+1} + a_n\) for \(n \geq 1\) and given initial conditions \(a_1 = 1\) and \(a_2 = 3\). Define another sequence \(b_n\) where \(b_n\) represents the number of distinct pairs \((x, y)\) of integers such that \(a_x + a_y = a_n\). Find \(b_{10}\).
**label:** `0`

### 17. (p̂=0.556, r_unc=-)
**Q:** Let \( S \) be a finite set of points in the plane. Determine the minimum number of colors required to color the points in \( S \) such that for any three distinct points \( A, B, C \in S \) forming a non-degenerate triangle, \( A \) and \( B \) have different colors if and only if \( B \) and \( C \) have the same color.
**label:** `2`

### 18. (p̂=0.333, r_unc=-)
**Q:** Let $P(x)$ be a polynomial with integer coefficients such that $P(0) = 1$, $P(1) = 2$, and $P(n) \equiv 0 \pmod{n}$ for all positive integers $n$. Find the minimum possible degree of $P(x)$, and determine the number of distinct polynomials $Q(x)$ with integer coefficients that satisfy $P(Q(x)) = x$ for all integers $x$. Compute the remainder when this number is divided by $1000$.
**label:** `1`

### 19. (p̂=0.556, r_unc=-)
**Q:** In a magic realm, there are n islands, each inhabited by a gnome. The gnome on island i holds 2^i gold coins, with i ranging from 1 to n. The gnomes can magically exchange any two sets of coins simultaneously; however, they can do this operation only once per day. Your task is to, using the minimum number of days, balance the total number of coins among all the islands so that each gnome has an equal number of coins. Let T(n) represent the minimum number of days required to achieve this balance for n islands. Determine T(n) for the general case and then compute T(10).
**label:** `9`

### 20. (p̂=0.444, r_unc=-)
**Q:** In the Cartesian plane, let P be a convex polygon with vertices at points with integer coordinates. Suppose that the area of P is an integer. Let L be the set of all lines parallel to either the x-axis or y-axis such that no point with integer coordinates lies on L and L separates P into two non-empty regions. Define the function f(P) to be the maximum number of lines in L that can be drawn such that no two intersect any vertex or edge of P more than once. Determine the greatest possible value of f(P) for any such polygon P with area 100.
**label:** `18`

### 21. (p̂=0.333, r_unc=-)
**Q:** Alice is given a set of integers from 1 to 100, inclusive. She defines a special operation on this set called a "Magic Move." A Magic Move involves selecting any three distinct integers \(a\), \(b\), and \(c\) from the set, and replacing them with the single integer \(a^2 + b^2 + c^2\). This operation can be performed any number of times.

Determine the minimum number of operations required to reduce the set to a single integer using only Magic Moves.
**label:** `50`

### 22. (p̂=0.333, r_unc=-)
**Q:** Alice has a set of \( n \) distinct positive integers \( \{a_1, a_2, \ldots, a_n\} \). She wants to select a subset of these integers such that the product of the elements in this subset is divisible by \( k \). Define \( f(n) \) as the minimum value of \( n \) such that there exists such a subset. Given \( k \) is a prime number, find the value of \( f(2024) \).
**label:** `2024`

### 23. (p̂=0.333, r_unc=-)
**Q:** Alice and Bob play a game on a circular table with 2024 evenly spaced points numbered from 1 to 2024. Initially, Alice places a marker on point 1. Then, they take turns placing markers on unoccupied points, moving clockwise around the table. The game ends when no more markers can be placed, either due to all points being occupied or because every possible move would result in a marker landing on an already occupied point. Determine the smallest positive integer \( n \) such that Alice, by strategically placing her marker, can guarantee that Bob is forced to place his marker on point \( n \).
**label:** `1012`

### 24. (p̂=0.556, r_unc=-)
**Q:** Consider a finite set of points in the plane, no three of which are collinear. Prove that there exists a point $P$ in the plane such that for any line $\ell$ through $P$, the number of points of the set lying above $\ell$ is equal to the number of points of the set lying below $\ell$.
**label:** `P`

### 25. (p̂=0.556, r_unc=-)
**Q:** Alice and Bob play a game on a complete graph with $2024$ vertices, each vertex labeled with a unique integer from $1$ to $2024$. Alice and Bob alternate turns, with Alice going first. Alice starts by choosing any vertex, and then each subsequent move consists of choosing a vertex that is not adjacent to the previous vertex chosen by the player. The game ends when a player is unable to make a move.

Alice's goal is to minimize the number of vertices that remain unselected. Bob's goal is to maximize this number. Assuming optimal play from both players, how many vertices will remain unselected when the game ends?
**label:** `0`

### 26. (p̂=0.444, r_unc=-)
**Q:** Alice and Bob are playing a game with a deck of cards numbered from 1 to 50. They take turns drawing cards from the deck without replacement. The game ends when the sum of the numbers on all drawn cards is exactly 1000. If Alice draws first, what is the probability that she draws the last card?
**label:** `\frac{1}{2}`

### 27. (p̂=0.556, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(0) = 1 \) and \( P(n) \) is a perfect square for all positive integers \( n \). Prove that there exists a polynomial \( Q(x) \) with integer coefficients such that \( P(x) = Q(x^2) \) for all \( x \).
**label:** `Q(x)`

### 28. (p̂=0.667, r_unc=-)
**Q:** A bag initially contains 2023 balls numbered from 1 to 2023. Two players take turns, removing one ball at a time, and keep track of the sum of the numbers on their removed balls. If a player can choose to end the game at any point, the player must also end it if they have a ball whose number is congruent to the sum of all removed numbers so far modulo 2023. If no one can win, the game is a tie. What is the minimum number of turns required to guarantee that the game will end? Assume optimal play from both players.
**label:** `2023`

### 29. (p̂=0.556, r_unc=-)
**Q:** In the realm of complex numbers, let $P(z)$ be a monic polynomial of degree 100 with integer coefficients such that $P(1) = 2^{100}$. Suppose further that $P(z)$ has exactly 50 distinct roots, all of which are non-real and occur in conjugate pairs. Let $S$ be the set of all possible values of $P(0)$. Determine the number of distinct prime factors of the largest element in $S$.
**label:** `1`

### 30. (p̂=0.778, r_unc=-)
**Q:** We are given a sequence of real numbers $(a_n)_{n\geq1}$ such that $a_1 = 1$ and for each $n \geq 1$, $a_{n+1}$ is chosen to be a random real number uniformly distributed in the interval $(0, a_n)$. Let $p$ be the probability that there exists a positive integer $k$ such that $a_k < \frac{1}{10}$. Find the smallest integer greater than $p$.
**label:** `2`

### 31. (p̂=0.444, r_unc=-)
**Q:** Aldo plays a card game where he draws cards from a standard deck of 52 cards. He draws a set of cards and immediately notices that all of his cards have even values if their faces are valued as follows: A=1, 2=2, 3=3, 4=4, 5=5, 6=6, 7=7, 8=8, 9=9, 10=10, J=11, Q=12, K=13. How many cards, on average, will Aldo draw in order to obtain at least one card with an even value?
**label:** `\frac{13}{6}`

### 32. (p̂=0.667, r_unc=-)
**Q:** Alice and Bob play a game with an infinite sequence of integers, starting with Alice's turn. The game has the following rules:
1. Each player can choose an integer i (i ≥ 0) and replace the current sequence with a new sequence obtained by removing all integers in positions congruent to i modulo (i+1) from the original sequence. 
2. After their turn, a player takes the difference between the largest and smallest remaining numbers and records it.
3. The game continues until only one integer remains in the sequence.
Alice's objective is to minimize the sum of differences she records throughout the game. Bob's objective is to maximize this sum.
Given an initial sequence, determine the sum of differences that would occur if both players play optimally.
**label:** `0`

### 33. (p̂=0.667, r_unc=-)
**Q:** Let \( f(x) \) be a polynomial with integer coefficients such that \( f(1) = 3 \), \( f(2) = 7 \), and for all integers \( n \geq 3 \), \( f(n) \equiv 0 \pmod{n(n-1)} \). Given that \( f(x) \) has exactly two non-real roots, find the smallest possible degree of \( f(x) \).
**label:** `4`

### 34. (p̂=0.333, r_unc=-)
**Q:** Let $P(x)$ be a polynomial with integer coefficients such that $P(1) = 3$ and $P(2) = 5$. Define a sequence of integers $\{a_n\}$ by $a_1 = 1$ and $a_{n+1} = P(a_n)$ for $n \geq 1$. Find all possible values of $a_{2024}$.
**label:** `3`

### 35. (p̂=0.556, r_unc=-)
**Q:** Determine the smallest integer n ≥ 5 such that for all sets S with exactly n elements there exist two disjoint subsets of S, namely T and U, satisfying |T|+|U|=n and ∀t∈T, ∃u∈U with u=2t or 2t=-1.
**label:** `6`

### 36. (p̂=0.667, r_unc=-)
**Q:** Let $P(x)$ be a polynomial with integer coefficients satisfying $P(1) = 5$, $P(2) = 7$, and $P(3) = 11$. Suppose that for all integers $n \geq 4$, $P(n)$ is a prime number. Find the largest possible value of $P(4)$.
**label:** `17`

### 37. (p̂=0.667, r_unc=-)
**Q:** Alice and Bob play a game on an infinite number line. Initially, Alice is at position 0. In each turn, Alice can move either +1 or -1 on the number line. Bob starts by picking an integer k &gt; 0. Alice aims to avoid visiting any integer of the form \(n^k\) for any positive integer \(n\), while Bob wants Alice to visit as many such numbers as possible. Assuming both play optimally, what is the smallest integer \(k\) such that Bob can force Alice to visit at least one integer of the form \(n^k\) within the first 100 moves?
**label:** `2`

### 38. (p̂=0.667, r_unc=-)
**Q:** Let \(P(x)\) be a polynomial with integer coefficients such that \(P(1) = P(2) = \ldots = P(2023) = 2024\). Suppose further that \(P(0)\) is a prime number. Determine the minimum number of positive divisors that \(P(n)\) can have for any integer \(n\), given that \(n\) is not one of the values for which \(P(n) = 2024\). Additionally, prove that such a polynomial exists.
**label:** `2`

### 39. (p̂=0.556, r_unc=-)
**Q:** Let $P(x)$ be a monic polynomial with integer coefficients such that $P(0) = 1$ and $P(1) = 2^n$ for some positive integer $n$. Suppose there exist distinct integers $a_1, a_2, \ldots, a_{n}$ such that $P(a_i) = 0$ for $i = 1, 2, \ldots, n$. Find the minimum possible value of $n$.
**label:** `4`

### 40. (p̂=0.556, r_unc=-)
**Q:** Alice and Bob play a game on an infinite number line. Alice starts at 0 and Bob starts at 100. They take turns rolling a standard six-sided die and moving their token right by the number of spaces rolled. If Alice and Bob land on the same space simultaneously, Alice wins. Otherwise, the game continues indefinitely. What is the probability that Alice wins on her third turn?
**label:** `0`


## iter 5  (N=4564, showing 40)

### 1. (p̂=0.444, r_unc=-)
**Q:** Let \( S \) be a set of \( n \) points in the plane, no three of which are collinear. A "tripod" is defined as a triple of distinct points \( (A, B, C) \) from \( S \) such that the circumcircle of triangle \( ABC \) contains exactly one additional point from \( S \) inside it. What is the minimum value of \( n \) for which it's guaranteed that there are at least 1000 different tripods in \( S \)?
**label:** `20`

### 2. (p̂=0.667, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(0) = 1 \), \( P(1) = 2 \), and \( P(n) \) is a perfect square for all integers \( n \geq 2 \). Prove that there exists an integer \( m \) such that \( P(m) \) is a perfect square and \( |m| > 1 \).
**label:** `2`

### 3. (p̂=0.778, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(1) = P(2) = \cdots = P(2024) = 2025 \). Define \( Q(x) = x^{2024}P(x) + 1 \). Suppose \( Q(x) \) has a complex root \( z \) with \( |z| = 1 \) and \( \arg(z) \) is a rational multiple of \( \pi \). What is the minimum possible degree of \( P(x) \)?
**label:** `2024`

### 4. (p̂=0.444, r_unc=-)
**Q:** Alice and Bob play a game on an infinite grid of integer coordinates. They start with a token at the origin (0, 0). On each turn, Alice moves the token to a point (x + a, y + b), where a and b are integers such that |a| + |b| = k, with k being the number of the turn (k = 1, 2, 3, ...). Bob aims to trap the token in a finite region of the grid that does not contain the origin by blocking certain points. What is the minimum number of turns n needed to ensure Bob can block enough points so that Alice cannot escape to infinity?
**label:** `3`

### 5. (p̂=0.667, r_unc=-)
**Q:** Let $P(x)$ be a monic polynomial of degree $n$ with integer coefficients, and let $Q(x)$ be another monic polynomial of degree $n$ such that $Q(x) = P(x)P(-x)$. Suppose further that $Q(x)$ has $n$ distinct integer roots. Find the smallest possible value of $n$ for which there exists such a pair $(P, Q)$.
**label:** `4`

### 6. (p̂=0.556, r_unc=-)
**Q:** A set of 2023 points is placed on the circumference of a circle, splitting it into 2023 arcs. Every point $P$ is assigned the value $\displaystyle{\frac{p-1}{p}}$, where $p$ is the number of prime factors of the length of the arc directly clockwise to $P$. For each arc, its value is determined by the geometric mean of the points on each side of that arc. Let $x$ be the minimum value over all arcs. What is the smallest value $k$ for which it is impossible for there to exist a set of points such that $x=k$?
**label:** `\frac{1}{2}`

### 7. (p̂=0.667, r_unc=-)
**Q:** In the set \( S = \{1, 2, 3, \dots, 100\} \), a subset \( T \) of size 50 is chosen randomly. For each number \( x \) in \( T \), let \( f(x) \) be the smallest integer \( y \) such that \( y \ge x \) and \( y \) is relatively prime to all numbers in \( T \setminus \{x\} \). Compute the probability that the product \( f(1) \cdot f(2) \cdot \dots \cdot f(50) \) is a perfect square.
**label:** `\frac{1}{2}`

### 8. (p̂=0.375, r_unc=-)
**Q:** Let \( P(x) \) be a monic polynomial with integer coefficients such that \( P(1) = 2023 \) and \( P(n) \) is a perfect square for all integers \( n \geq 2 \). Suppose further that the roots of \( P(x) \) are all real and distinct. Find the minimum possible degree of \( P(x) \) and provide a polynomial that meets these conditions. If such a polynomial does not exist, prove it.
**label:** `2`

### 9. (p̂=0.444, r_unc=-)
**Q:** Alice and Bob are playing a game on a 2024×2024 grid. Initially, the cell at the bottom-left corner of the grid, denoted as (1,1), is filled with the number 1. Bob then writes the product of the numbers in the two neighbouring cells into each other cell of the grid that is adjacent to it in both horizontal and vertical directions. For instance, if the cell (1,2) is numbered x and the cell (2,1) is numbered y, then Bob writes xy into the cell (2,2). Bob has been writing products until he can no longer find a cell that satisfies the product requirement.

At the end of this game, Alice determines whether the sum of all numbers written on the grid is divisible by the number written in the top-right cell, denoted as (2024,2024). What is the minimum sum Alice must calculate to make it possible for the sum to be divisible by (2024,2024)?
**label:** `4096576`

### 10. (p̂=0.444, r_unc=-)
**Q:** Alice and Bob play a game on an infinite grid of unit squares. The game begins with a coin placed at the origin. On each turn, a player can perform one of two operations: either move the coin one square to the right or one square up, or "flip" the coin to the opposite side (from heads to tails or from tails to heads). The game ends when the coin lands on a square that has already been visited. Alice moves first, and the goal is to force Bob into ending the game in the fewest possible moves. Determine the minimum number of moves Bob can be forced to make, no matter how Alice plays.
**label:** `3`

### 11. (p̂=0.444, r_unc=-)
**Q:** Alice and Bob play a game on an infinite number line. Alice starts at 0 and may move to any integer or rational number $x$ such that $|x|<100$. Bob always moves by an integer amount. Each player moves at least once, and Alice's move occurs after Bob's. After Alice's move, both are trapped at their positions. They play optimally to maximize the total distance they travel before reaching their final positions. Determine the minimum distance Alice needs to move, if such a move exists, so that Alice and Bob together travel as far as possible.
**label:** `99`

### 12. (p̂=0.556, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(1) = 2023 \) and for all integers \( n \geq 2 \), \( P(n) \) is the smallest positive integer not yet in the sequence \( P(1), P(2), \ldots, P(n-1) \). Determine the maximum number of consecutive terms in the sequence \( P(1), P(2), \ldots, P(10^6) \) that can form an arithmetic progression.
**label:** `1`

### 13. (p̂=0.444, r_unc=-)
**Q:** Let $P$ be a convex $n$-gon in the plane, where $n \geq 4$. Suppose that for every set of three non-collinear vertices $A$, $B$, and $C$, there exists a circle passing through $A$, $B$, and $C$. Let $c(n)$ be the smallest integer such that there exists a coloring of the vertices of $P$ using $c(n)$ colors such that no three vertices of the same color are collinear and no three vertices of the same color lie on the same circle. Determine the value of $c(n)$ for $n = 2024$.
**label:** `2`

### 14. (p̂=0.444, r_unc=-)
**Q:** Find the smallest integer $n \geq 2$ such that for all integers $a$ and $b$ with $1 \leq a, b \leq n$, the expression
\[
\frac{a^n - 1}{a - 1} + \frac{b^n - 1}{b - 1}
\]
is divisible by $(a + b)^{2024}$.
**label:** `2024`

### 15. (p̂=0.444, r_unc=-)
**Q:** Determine all real polynomials \( Q(x) \) for which the expression \( \frac{Q(n)}{n^2} \) approaches zero as \( n \) approaches infinity, and yet for some constant \( c > 0 \), there exist infinitely many integers \( k \) such that \( Q(k) \ge c k^2 \).
**label:** `0`

### 16. (p̂=0.333, r_unc=-)
**Q:** Alice and Bob are playing a game on an infinite grid, where each cell is either white or black. The game starts with a finite arrangement of black cells, and the rest are white. On each turn, a player selects a square of side length \(n\) (\(n \geq 1\)) and flips the color of every cell within this square. Alice moves first. The game ends when it is impossible for either player to make a move that changes the state of any cell. Determine the smallest positive integer \(n\) for which Alice has a winning strategy that guarantees the number of black cells will eventually exceed a given threshold of 100.
**label:** `2`

### 17. (p̂=0.778, r_unc=-)
**Q:** Let $P(x)$ be a monic polynomial of degree $n$ with integer coefficients such that $P(1) = 2023$ and $P(2) = P(3)$. Find the minimum possible value of $n$.
**label:** `2`

### 18. (p̂=0.556, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(0) = 1 \), \( P(1) = 2 \), and for any positive integer \( n \), \( P(n) \) divides \( 2^n - 1 \). Define a sequence \( \{a_n\} \) by \( a_0 = 1 \) and \( a_{n+1} = P(a_n) \) for \( n \geq 0 \). If \( S \) is the set of all positive integers \( k \) for which \( a_k \) is prime, what is the smallest positive integer \( m \) such that for any prime \( p \) not in \( S \), there exists a \( k \in S \) for which \( p \) divides \( a_k \)?
**label:** `1`

### 19. (p̂=0.778, r_unc=-)
**Q:** Let $P(x)$ be a polynomial with integer coefficients such that $P(1) = P(2) = \dots = P(n) = 1$ for some integer $n \geq 2$. Define $Q(x)$ as the polynomial with integer coefficients satisfying $Q(1) = Q(2) = \dots = Q(n) = 2$ and $Q(0) = -1$. Determine the smallest possible value of $n$ for which there exists an integer $k$ such that $P(x) \cdot Q(x) = R(x)^2 + k$ for some polynomial $R(x)$ with integer coefficients.
**label:** `2`

### 20. (p̂=0.444, r_unc=-)
**Q:** Alice and Bob are playing a game on a table with \( n \) circular coins placed on it, each with radius \( r \). The coins can be placed anywhere on the table such that no two coins overlap. On Alice's turn, she selects a coin and flips it over, moving it to a position such that its center is exactly distance \( d \) from the center of the coin that was flipped over. Bob's turn then follows, with the same rules as Alice. The game proceeds until no valid moves remain. Alice wins if there exists at least one move where Bob cannot make a move in response; otherwise, Bob wins. Determine all values of \( n \) for which Alice can force a win for some given \( r \) and \( d \).
**label:** `1`

### 21. (p̂=0.667, r_unc=-)
**Q:** Let $P(x) = x^3 - 3x^2 + 4x - 2$ and $Q(x) = x^4 - 2x^3 + 3x^2 - 4x + 5$. If $r$ is a real root of $P(x)$ and $s$ is a complex root of $Q(x)$ such that $s$ and $\overline{s}$ are conjugates, find the minimum possible value of $|r + s|$. Express your answer as a fraction in lowest terms.
**label:** `1`

### 22. (p̂=0.333, r_unc=-)
**Q:** Find all ordered triples of real numbers (x, y, z) satisfying the following system of equations: x³ - y = 1, y³ - z = 1, z³ - x = 1.
**label:** `(1, 1, 1)`

### 23. (p̂=0.778, r_unc=-)
**Q:** Alice and Bob play a game on an infinite grid of integers. Initially, every integer cell is white. Alice and Bob take turns painting cells black; on a turn, a player must paint exactly 2024 cells black. Alice goes first and may paint any cells black on her first turn. After that, each player must always paint a cell black that is adjacent to at least one other cell that is already black. The game continues until all cells on the grid are painted black. Alice wins if, at the end of the game, there exists an integer \varphi such that any two black cells on the grid share an edge and their centers are in a horizontal or vertical distance of at most \varphi from each other. Otherwise, Bob wins.

What is the smallest integer \varphi for which Alice has a winning strategy?
**label:** `2023`

### 24. (p̂=0.444, r_unc=-)
**Q:** Let \( P(x) \) be a monic polynomial with integer coefficients such that \( P(0) \) and \( P(1) \) are both odd primes. Suppose further that for all positive integers \( n \), the value \( P(n) \) can be expressed uniquely as the product of two distinct positive integers greater than 1. Determine the minimum possible degree of \( P(x) \).
**label:** `3`

### 25. (p̂=0.444, r_unc=-)
**Q:** Find all continuous functions \( f: \mathbb{R} \to \mathbb{R} \) such that for all real numbers \( x \) and \( y \), the equation \( f(x + y) + f(x - y) = 2f(x)f(y) \) holds, and additionally, \( \lim_{x \to 0} \frac{f(x)}{x} = 1 \).
**label:** `f(x) = x`

### 26. (p̂=0.444, r_unc=-)
**Q:** In the kingdom of Mathesia, there exists an infinite grid of unit squares, where each square is colored either black or white. Initially, all squares are white except for a single black square at position (0, 0). Every minute, a black square turns into a white square if and only if the average of the colors of the neighboring (horizontally and vertically adjacent) unit squares is strictly less than \( \frac{3}{4} \). A square is colored black otherwise. Let \( B_n \) be the set of all black squares at time \( n \), and let \( A_n \) be the maximum side length of any axis-parallel square entirely contained in \( B_n \). Find the minimum value of \( A_{2024} \).
**label:** `1`

### 27. (p̂=0.556, r_unc=-)
**Q:** Alice and Bob are playing a game on a $2024 \times 2024$ grid. Alice starts at the bottom-left corner, while Bob starts at the top-right corner. On each turn, Alice can move right, up, or diagonally up-right (to her top-right neighbor), and Bob can move left, down, or diagonally down-left (to his bottom-left neighbor). They move simultaneously, and each can see the opponent's moves. The game ends when either reaches the other's starting position for the first time. What is the probability that the game ends in exactly 2024 turns? How should Alice and Bob coordinate their moves so as to increase this probability?
**label:** `1`

### 28. (p̂=0.778, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(0) = 1 \) and \( P(1) = 2 \). Suppose further that for any integer \( n \), \( P(n) \) is a perfect square if and only if \( n \) is a perfect square. Determine the smallest possible degree of \( P(x) \) and find an explicit form for such a polynomial.
**label:** `2`

### 29. (p̂=0.556, r_unc=-)
**Q:** Suppose you have a finite grid of points in the Cartesian plane where no three points are collinear. A path on this grid consists of a sequence of moves from one point to another with each move being one unit along a row or column. Given that for any point on this grid, there exists a path from it to every other point on the grid using no more than 10 moves, what is the minimum number of rows and columns this grid can have?
**label:** `6`

### 30. (p̂=0.444, r_unc=-)
**Q:** Alice and Bob play a game on an infinite chessboard. Alice starts by placing a pawn on any square. Each subsequent move consists of Bob choosing a positive integer distance \(k\) (where \(k\) can be 1 or 2), and then Alice moving the pawn to any square on the chessboard that is exactly \(k\) squares away horizontally, vertically, or diagonally from its current position. The game continues until Alice can no longer move the pawn to an unoccupied square within the chosen distance. Bob wins if the game ends in fewer than \(n\) moves; otherwise, Alice wins.

What is the minimum integer \(n\) for which Bob has a winning strategy?
**label:** `3`

### 31. (p̂=0.444, r_unc=-)
**Q:** There exists a sequence of distinct real numbers {a1, a2, ..., a10} such that each element ai (i = 1, 2, ..., 10) satisfies the following properties:

1) ai is between 0 and 1, i.e., 0 < ai < 1;
2) ai * (1 - ai) is not the smallest element of the sequence among the products ai * (1 - ai) for all i;
3) For any two distinct elements ai and aj (i ≠ j), ai + aj ≠ 1;
4) There exists exactly one element ai for which there is another element aj satisfying both of the following:
   a) |ai - aj| ≤ 0.1;
   b) |ai^2 + aj^2 - 1| < 0.01.

Find all possible values for the smallest element a1 in the sequence {a1, a2, ..., a10}, and let S denote their sum. Determine the value of ∑_{k=1}^{S} k^3.
**label:** `1`

### 32. (p̂=0.444, r_unc=-)
**Q:** Alice and Bob play a game on an infinite grid of unit squares. Alice starts by choosing any square and marking it with an A. On each turn, the player adds X to the set S, where X is a set of all squares that share a side or a vertex with any square in S. If a square is added to S for the first time, the player gets a point. The game ends when S has infinitely many squares. The player with more points at the end of the game wins. Find the number of possible game outcomes.
**label:** `1`

### 33. (p̂=0.444, r_unc=-)
**Q:** Let $P(x)$ be a polynomial with integer coefficients, such that for some fixed integer $m \geq 2$, $P(x) \equiv 0 \mod m$ whenever $x \equiv 0 \mod m$. Determine the smallest possible value of $m$ for which there exists an integer $a$ coprime to $m$ such that the sequence $\{P(na) \mod m\}_{n \geq 0}$ contains at least one repeated element.
**label:** `2`

### 34. (p̂=0.778, r_unc=-)
**Q:** Let \(P(x)\) be a polynomial with integer coefficients such that \(P(1) = 2023\), and for any integer \(k\), \(P(k^2 + k) - P(k)\) is divisible by 2023. Find the smallest possible value of \(P(2023)\).
**label:** `2023`

### 35. (p̂=0.667, r_unc=-)
**Q:** Alice and Bob are playing a game on an infinite grid of unit squares. Each square is initially uncolored. On each turn, a player must choose an uncolored square and color it red, provided that after coloring, every colored square (including the one just colored) can be part of a connected path of red squares linking any two colored squares. A "connected path" means a series of horizontally or vertically adjacent red squares that starts and ends on a colored square.

Alice colors the first square (the bottom-left corner, which she colors red). After that, the players take turns. The game ends when it's no longer possible to color a square without breaking the rules. The player who colors the last possible square wins. If Alice and Bob use optimal strategies, who wins and what is their winning strategy?
**label:** `Alice`

### 36. (p̂=0.333, r_unc=-)
**Q:** Alice and Bob play a game on an infinite grid of points. Initially, Alice places a red stone on any point of the grid. Then, they take turns performing the following operation: choose a point that is at a distance of exactly 1, 2, or 3 units from the currently placed stone, and place a stone of their color on that point. Alice moves first, and both players know the exact position of each other's stones at all times. The goal is to be the first to place a stone of your color on a point that is equidistant from all three previously placed stones of the opposite color. Determine the smallest number of turns (including Alice's initial placement) after which Alice is guaranteed to win, regardless of Bob's strategy.
**label:** `4`

### 37. (p̂=0.333, r_unc=-)
**Q:** Alice and Bob are playing a game on a 100x100 grid of unit squares. Each turn, a player chooses a previously unselected square and marks it with their color (Alice's color is red, Bob's color is blue). The game ends when one player cannot make a move, and that player loses the game. Before the game starts, Alice and Bob choose a number k. Alice starts the game. She wins if there exists a rectangle on the grid of size at least kxk, whose perimeter is entirely colored red, and all other squares inside this rectangle are uncolored. Bob wins if this condition is not met. Determine all values of k for which Alice has a winning strategy.
**label:** `1`

### 38. (p̂=0.444, r_unc=-)
**Q:** Let \( P(x) \) be a monic polynomial with integer coefficients such that \( P(0) = 2024 \). Suppose that for every positive integer \( n \), the equation \( P(x) = n \) has exactly \( n \) distinct integer solutions. Find the smallest possible value of \( P(1) \).
**label:** `2025`

### 39. (p̂=0.778, r_unc=-)
**Q:** A group of \(n\) people are seated around a round table, where \(n \geq 2\). Each person is either honest (always tells the truth) or a liar (always lies). One of the people claims to be an honest person and that there is at least one liar at the table. Another person, who is adjacent to the first, claims to be a liar and that there is at least one honest person at the table. If the first two claims are contradictory, and if there are exactly \(k\) honest people at the table, where \(2 \leq k \leq n-1\), determine the smallest possible value of \(n\) for which such a scenario is possible.
**label:** `3`

### 40. (p̂=0.556, r_unc=-)
**Q:** On the Cartesian plane, let $A = (0, 0)$, $B = (4, 0)$, and $C = (4, 3)$. A point $D$ lies inside the triangle $ABC$ such that the distance from $D$ to each side of the triangle is equal. If the area of triangle $ABD$ is $2\sqrt{3}$ and the area of triangle $BCD$ is $2$, find the area of triangle $ACD$. Express your answer as a fraction in lowest terms.
**label:** `4 - 2\sqrt{3}`
