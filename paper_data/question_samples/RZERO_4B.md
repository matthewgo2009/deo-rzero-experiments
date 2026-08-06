# RZERO_4B — question samples


_R-Zero trained-questioner training set, Qwen3-4B (post-filter p̂∈[0.3,0.8])_

Per iteration: total pool size N, then an evenly-spaced sample of 40 questions.
Fields — `q`: question text; `label`: pseudo-label/answer used for training; `gt`: challenger-proposed answer (DEO only); `phat`: self-consistency p̂ = modal-count/m (R-Zero `score` is the same quantity); `runc`: uncertainty reward 1−2|p̂−0.5| (DEO only).


## iter 1  (N=3939, showing 40)

### 1. (p̂=0.778, r_unc=-)
**Q:** A square grid is formed by placing points at integer coordinates in the plane. A path is defined as a sequence of moves starting from the origin \((0,0)\) and ending at \((10,10)\) where each move is either to the right (increasing the x-coordinate by 1) or up (increasing the y-coordinate by 1). How many such paths are there that do not pass through the point \((5,5)\)?
**label:** `121252`

### 2. (p̂=0.778, r_unc=-)
**Q:** 在一个由n个点构成的简单图G中，每个点都连接到其他n-1个点上。如果我们要从这n个点中选择m个点（m<n），并且要求这m个点之间两两相连形成一个完全子图（即Km），请问有多少种不同的选择方式？假设n=10, m=5。
**label:** `252`

### 3. (p̂=0.333, r_unc=-)
**Q:** A square with side length 10 units is inscribed in a circle. A smaller square is then inscribed in the same circle such that its vertices touch the circle's circumference at points equidistant from the vertices of the larger square. What is the area of the smaller square?
**label:** `50`

### 4. (p̂=0.667, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial of degree 6 with integer coefficients such that \( P(1) = 2023 \) and \( P(2023) = 1 \). Prove that \( P(x) \) cannot have more than 6 real roots.
**label:** `6`

### 5. (p̂=0.778, r_unc=-)
**Q:** Let \(ABC\) be an acute-angled triangle with circumcircle \(\Gamma\). Let \(D, E,\) and \(F\) be the midpoints of arcs \(BC, CA,\) and \(AB\) respectively (not containing the opposite vertices). Lines \(AD, BE,\) and \(CF\) meet \(\Gamma\) again at points \(X, Y,\) and \(Z\) respectively. Prove that the triangle \(XYZ\) is similar to the triangle \(DEF\).
**label:** `\triangle XYZ \sim \triangle DEF`

### 6. (p̂=0.375, r_unc=-)
**Q:** In a tournament, 12 teams participate. Each team plays with every other team exactly once. The winner of each match receives 2 points, the loser receives 0 points, and in case of a draw, each team receives 1 point. At the end of the tournament, it is observed that no two teams have the same number of points. What is the minimum possible number of draws that occurred during the tournament?
**label:** `33`

### 7. (p̂=0.500, r_unc=-)
**Q:** In a sequence of positive integers, every term after the first two terms is the sum of the two previous terms. If the first term is 2 and the sixth term is 32, what is the second term?
**label:** `5`

### 8. (p̂=0.333, r_unc=-)
**Q:** Let $ABCDEF$ be a regular hexagon with side length $1$. A point $P$ is chosen inside the hexagon such that $\angle BAP = 30^\circ$ and $\angle FCP = 60^\circ$. Find the area of triangle $APF$.
**label:** `\frac{\sqrt{3}}{4}`

### 9. (p̂=0.333, r_unc=-)
**Q:** A sequence of numbers starts with 1 and each subsequent number is the sum of the squares of the digits of the previous number. For example, the first number is 1, the second number is \(1^2 = 1\), the third number is \(1^2 = 1\), and so on. Find the smallest positive integer that appears exactly four times in this sequence.
**label:** `1`

### 10. (p̂=0.444, r_unc=-)
**Q:** 在一个正方形棋盘上，每个格子都有一个不同的整数。假设这个正方形的边长是4格，共有16个小正方形。现在我们随机选择正方形上的一个点，这个点落在任何一个格子内部的概率相同。求这个随机点落在边长为2格的正方形内的概率是多少？
**label:** `0.25`

### 11. (p̂=0.444, r_unc=-)
**Q:** In the plane, let \( S \) be the set of all points \((x, y)\) such that \( x \) and \( y \) are integers and \( 1 \leq x \leq 2023 \), \( 1 \leq y \leq 2023 \). Let \( T \) be the set of all points in \( S \) that can be reached from \((1,1)\) by moving to adjacent points (up, down, left, or right) such that each move takes you to a point that has not been visited before. Determine the number of points in \( T \).
**label:** `2023 \times 2023`

### 12. (p̂=0.778, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(2) = 5 \) and \( P(3) = 10 \). What is the smallest possible value of \( P(1) \)?
**label:** `0`

### 13. (p̂=0.778, r_unc=-)
**Q:** Let \( a, b, c \) be positive real numbers such that \( a + b + c = 1 \). Prove that
\[ \frac{a}{1 + b^2} + \frac{b}{1 + c^2} + \frac{c}{1 + a^2} \leq \frac{9}{10}. \]
**label:** `\frac{9}{10}`

### 14. (p̂=0.333, r_unc=-)
**Q:** In a magical forest, there are three types of trees: oak, pine, and willow. The number of oak trees is twice the number of pine trees, and the number of willow trees is three times the number of pine trees. If the total number of trees in the forest is 120, how many of each type of tree are there?
**label:** `60`

### 15. (p̂=0.778, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the sum of the first \( n \) terms of the arithmetic sequence \( 2, 7, 12, \ldots \) is a perfect square.
**label:** `2`

### 16. (p̂=0.333, r_unc=-)
**Q:** Let \( ABCD \) be a convex quadrilateral with \( AB = 5 \), \( BC = 6 \), \( CD = 7 \), \( DA = 8 \). If the diagonals \( AC \) and \( BD \) intersect at point \( P \) such that \( AP : PC = 2 : 3 \) and \( BP : PD = 3 : 4 \), find the area of quadrilateral \( ABCD \).
**label:** `42`

### 17. (p̂=0.778, r_unc=-)
**Q:** What is the smallest positive integer \( n \) such that \( n^2 + 1 \) is divisible by both 2 and 5?
**label:** `3`

### 18. (p̂=0.444, r_unc=-)
**Q:** Let \(a\), \(b\), and \(c\) be positive real numbers such that \(abc = 1\). Prove that:
\[
\frac{a^2 + b^2 + c^2}{ab + bc + ca} + \frac{8}{a + b + c} \geq 5.
\]
**label:** `5`

### 19. (p̂=0.556, r_unc=-)
**Q:** In the sequence defined by \( a_1 = 1 \) and \( a_{n+1} = a_n + \frac{1}{a_n} \), find the integer part of \( a_{100} \).
**label:** `14`

### 20. (p̂=0.444, r_unc=-)
**Q:** Let \( ABCD \) be a convex quadrilateral with \( AB = 5 \), \( BC = 6 \), \( CD = 7 \), and \( DA = 8 \). Diagonals \( AC \) and \( BD \) intersect at point \( E \) such that the area of triangle \( AEB \) is 10. Find the length of diagonal \( AC \).
**label:** `10`

### 21. (p̂=0.444, r_unc=-)
**Q:** What is the smallest positive integer \( n \) such that the polynomial \( P(x) = x^n + ax^{n-1} + bx^{n-2} + \cdots + cx + d \) with integer coefficients satisfies \( P(1) = 10 \) and \( P(2) = 20 \)?
**label:** `2`

### 22. (p̂=0.333, r_unc=-)
**Q:** In the sequence of all positive integers, the numbers that are multiples of 7 are removed. The next number in the sequence that is a perfect square is then replaced with its square root. What is the 100th number in this modified sequence?
**label:** `100`

### 23. (p̂=0.444, r_unc=-)
**Q:** In the coordinate plane, consider the ellipse defined by the equation \(\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1\) where \(a > b > 0\). A point \(P\) on the ellipse is such that the tangents to the ellipse at \(P\) intersect the \(x\)-axis at points \(A\) and \(B\) and the \(y\)-axis at points \(C\) and \(D\). If the area of the quadrilateral formed by \(A\), \(B\), \(C\), and \(D\) is \(128\) square units, find the value of \(a + b\).
**label:** `16`

### 24. (p̂=0.333, r_unc=-)
**Q:** Let \( f(x) \) be a polynomial of degree 4 such that \( f(1) = 1, f(2) = 4, f(3) = 9, f(4) = 16, \) and \( f(5) = 25. \) Find the value of \( f(6). \)
**label:** `36`

### 25. (p̂=0.500, r_unc=-)
**Q:** What is the smallest positive integer \( n \) such that there exist integers \( a, b, c, d \) with \( 1 < a < b < c < d \) and \( a + b + c + d = n \), where \( a^2 + b^2 + c^2 + d^2 \) is a perfect square?
**label:** `17`

### 26. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation \( x^2 + y^2 = z^2 \) has exactly \( n \) distinct solutions in positive integers \( x, y, z \) where \( x, y, z \) are all less than or equal to 100.
**label:** `16`

### 27. (p̂=0.375, r_unc=-)
**Q:** Let \( f(x) \) be a polynomial with real coefficients such that \( f(0) = 1 \) and \( f(x) \cdot f(1 - x) = 1 + x^2 \) for all real \( x \). Find the sum of the squares of the roots of \( f(x) \).
**label:** `2`

### 28. (p̂=0.375, r_unc=-)
**Q:** In a magical kingdom, there are three types of mystical creatures: unicorns, dragons, and griffins. Unicorns are known for their ability to heal any wound, dragons for their breath of fire, and griffins for their strength. The king of the kingdom declares that each creature will be assigned a number of gold coins based on the following rules:
1. The total number of gold coins for all creatures combined is 777.
2. The number of gold coins for each type of creature forms an arithmetic sequence.
3. If the number of gold coins for unicorns is 200, find the number of gold coins for dragons and griffins.
**label:** `318`

### 29. (p̂=0.333, r_unc=-)
**Q:** Let \( f(x) \) be a polynomial of degree 4 such that \( f(0) = 1 \), \( f(1) = 2 \), \( f(2) = 5 \), \( f(3) = 10 \), and \( f(4) = 17 \). Find the value of \( f(5) \).
**label:** `26`

### 30. (p̂=0.444, r_unc=-)
**Q:** Find the value of \( x \) such that the area of the triangle formed by the points \((0,0)\), \((x, 2x)\), and \((x, 0)\) is equal to 18 square units.
**label:** `3\sqrt{2}`

### 31. (p̂=0.556, r_unc=-)
**Q:** Let $ABC$ be an isosceles triangle with $AB = AC$ and $\angle BAC = 20^\circ$. Point $D$ lies on $AC$ such that $AD = BC$. Determine the measure of $\angle DBC$.
**label:** `30^\circ`

### 32. (p̂=0.444, r_unc=-)
**Q:** Let \(f(x)\) be a polynomial with integer coefficients such that \(f(1) = 10\), \(f(2) = 20\), \(f(3) = 30\), \(f(4) = 40\), and \(f(5) = 50\). Find the remainder when \(f(2023)\) is divided by 1000.
**label:** `230`

### 33. (p̂=0.500, r_unc=-)
**Q:** In a magical kingdom, there are three types of mystical creatures: unicorns, dragons, and phoenixes. Each creature has a unique property: unicorns heal, dragons create fire, and phoenixes can resurrect. A wizard is tasked with creating a spell that can summon exactly one of each creature. However, the spell must be cast in a sequence where no two consecutive spells summon creatures with opposite properties (for example, a unicorn cannot be immediately followed by a phoenix). If the wizard has already cast a spell summoning a unicorn, how many different sequences of spells can he cast to summon one of each creature?
**label:** `2`

### 34. (p̂=0.444, r_unc=-)
**Q:** Let \( S \) be the set of all real numbers \( x \) such that the equation \( x^2 + y^2 + z^2 = 2xy + 2yz + 2zx \) holds for some integers \( y \) and \( z \). Determine the number of elements in the set \( S \).
**label:** `\infty`

### 35. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation \( x^2 + y^2 + z^2 + w^2 = n \) has exactly 12 solutions in non-negative integers \( x, y, z, \) and \( w \).
**label:** `5`

### 36. (p̂=0.333, r_unc=-)
**Q:** In a magical kingdom, there are 2023 wizards, each possessing a unique spell. Every day, each wizard casts a spell, and the spells combine in a mysterious way to create a new spell. The new spell is created by combining the spells of the wizard with his neighbor to the left and right. However, on the day of a full moon, all wizards combine their spells to create a super spell. If the super spell is not created, the kingdom suffers a great misfortune. Assuming that the wizards always cast spells in a circular arrangement, what is the minimum number of full moons that must pass before the super spell is guaranteed to be created?
**label:** `2023`

### 37. (p̂=0.556, r_unc=-)
**Q:** What is the smallest positive integer \( n \) such that \( n! \) (the factorial of \( n \)) is divisible by \( 10^{100} \)?
**label:** `405`

### 38. (p̂=0.375, r_unc=-)
**Q:** 在一个正六边形内，每个顶点上放置一个点，这些点与该六边形的对角线相连形成一个小三角形。已知这些小三角形的面积之和等于大六边形面积的三分之一。如果大六边形的边长为1，那么这些小三角形顶点处所放的点与大六边形中心的距离之和是多少？
**label:** `3\sqrt{3}`

### 39. (p̂=0.625, r_unc=-)
**Q:** In a magical land, there are three types of coins: gold, silver, and bronze. A gold coin is worth 5 silver coins, and a silver coin is worth 7 bronze coins. If you have 1 gold coin, 2 silver coins, and 3 bronze coins, how many bronze coins would you have if you exchanged all your coins?
**label:** `52`

### 40. (p̂=0.667, r_unc=-)
**Q:** Let $ABC$ be an acute-angled triangle with circumcircle $\omega$. Let $M$ and $N$ be the midpoints of $AB$ and $AC$, respectively. The line passing through $M$ and $N$ intersects the circle $\omega$ at points $D$ and $E$ such that $D$ lies on arc $BC$ not containing $A$, and $E$ lies on arc $BC$ containing $A$. Let $F$ be the intersection of lines $AD$ and $BC$, and let $G$ be the intersection of lines $AE$ and $BC$. Prove that $FG$ is perpendicular to the line $BC$.
**label:** `FG \perp BC`


## iter 2  (N=4312, showing 40)

### 1. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that there exists a polynomial \( P(x) \) of degree \( n \) with integer coefficients that satisfies the equation \( P(1) = 1, P(2) = 2, P(3) = 3, \ldots, P(10) = 10 \).
**label:** `10`

### 2. (p̂=0.556, r_unc=-)
**Q:** In a triangular park, there are three paths, each connecting a different pair of vertices of the triangle. A person starts at one vertex and walks along the paths, taking exactly 4 steps. Each step is taken along one of the paths, and the person may use the same path more than once. However, the person cannot visit the same vertex more than once (excluding the starting vertex). How many distinct paths can the person take?
**label:** `6`

### 3. (p̂=0.444, r_unc=-)
**Q:** Find all positive integers \( n \) such that the equation \( x^2 + y^2 = z^2 + n \) has infinitely many solutions in positive integers \( x, y, \) and \( z \).
**label:** `0`

### 4. (p̂=0.778, r_unc=-)
**Q:** What is the smallest positive integer \( n \) such that the polynomial \( P(x) = x^n + x^{n-1} + \cdots + x + 1 \) has no integer roots?
**label:** `2`

### 5. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^n + x^{n-1} + \cdots + x + 1 \) is divisible by \( x^2 - 3x + 2 \).
**label:** `3`

### 6. (p̂=0.333, r_unc=-)
**Q:** Let \( S \) be a finite set of positive integers. Define \( f(S) \) as the sum of all elements in \( S \), and \( g(S) \) as the sum of the squares of all elements in \( S \). If \( |S| = n \), find the smallest possible value of \( g(S) - f(S)^2 \). 

Consider that \( S = \{a_1, a_2, \ldots, a_n\} \) where \( a_i \) are distinct positive integers and the sum \( f(S) = \sum_{i=1}^{n} a_i \). The problem requires minimizing \( g(S) - f(S)^2 \).
**label:** `-4`

### 7. (p̂=0.333, r_unc=-)
**Q:** A rectangular prism has dimensions 10 cm, 8 cm, and 6 cm. Inside this prism, a smaller rectangular prism is inscribed such that one of its faces is a diagonal square. What is the maximum possible volume of this inscribed rectangular prism?
**label:** `984`

### 8. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^4 - nx^3 + (n-1)x^2 - nx + 1 \) has at least one integer root.
**label:** `1`

### 9. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^n - 3x^{n-1} + 2x^{n-2} - \cdots + (-1)^{n-1} \cdot 2x - 1 \) has all its roots as integers.
**label:** `1`

### 10. (p̂=0.556, r_unc=-)
**Q:** In a game involving a set of 100 distinct cards numbered from 1 to 100, each player draws a card without replacement. The game ends when the sum of all drawn cards becomes greater than 10,000. What is the maximum number of cards that can be drawn such that the game still ends before drawing the next card?
**label:** `140`

### 11. (p̂=0.444, r_unc=-)
**Q:** Find the number of ordered pairs of integers $(x, y)$ that satisfy the equation $x^2 + y^2 + 2xy + 3x + 3y + 2 = 0$.
**label:** `\infty`

### 12. (p̂=0.556, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(1) = 5 \) and \( P(2) = 11 \). Find the smallest possible positive value of \( P(3) \).
**label:** `17`

### 13. (p̂=0.333, r_unc=-)
**Q:** A circle of radius 5 cm is inscribed in a square. A smaller circle is then inscribed in each corner of the square, such that each smaller circle is tangent to two sides of the square and the larger circle. Find the total area of all the smaller circles combined.
**label:** `25\pi`

### 14. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that \( n^2 + 15n + 56 \) is a perfect square.
**label:** `1`

### 15. (p̂=0.333, r_unc=-)
**Q:** A sequence of positive integers \( a_1, a_2, \ldots, a_{10} \) is called "balanced" if the sum of any two adjacent terms is a perfect square. Find the number of balanced sequences where \( a_1 = 1 \) and \( a_{10} = 16 \).
**label:** `2`

### 16. (p̂=0.333, r_unc=-)
**Q:** Find all pairs of integers \((x, y)\) such that \(x^2 + y^2 = 5(x + y)\).
**label:** `(0, 0), (0, 5), (5, 0), (5, 5)`

### 17. (p̂=0.778, r_unc=-)
**Q:** In a triangle ABC, the angle bisector of ∠A intersects BC at D. If the length of BD is twice the length of DC and the area of triangle ABC is 36 square units, find the area of triangle ABD.
**label:** `24`

### 18. (p̂=0.667, r_unc=-)
**Q:** What is the smallest positive integer n such that n! (n factorial) is divisible by 2^3 * 3^2 * 5^1 * 7^0 * 11^1 * 13^0 * 17^0 * 19^1?
**label:** `19`

### 19. (p̂=0.333, r_unc=-)
**Q:** A triangle ABC is inscribed in a circle with center O. The circle's radius is 10 units. If the length of side BC is 12 units and the angle bisector of angle A intersects BC at point D, what is the length of segment AD?
**label:** `8`

### 20. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 + x^2 + x + n \) has three distinct positive integer roots.
**label:** `6`

### 21. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^n - x^{n-1} - x^{n-2} - \cdots - x - 1 \) has all real roots.
**label:** `2`

### 22. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^n + a_{n-1}x^{n-1} + \cdots + a_1x + a_0 \) with integer coefficients has the property that \( P(k) \) is divisible by \( k \) for all positive integers \( k \) from 1 to \( n \).
**label:** `3`

### 23. (p̂=0.667, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the product of the first \( n \) positive integers (i.e., \( n! \)) is divisible by \( 2^{10} \) but not by \( 2^{11} \).
**label:** `12`

### 24. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^n - x^{n-1} - x^{n-2} - \cdots - x - 1 \) is divisible by \( x^2 - x - 1 \).
**label:** `6`

### 25. (p̂=0.556, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(1) = 2 \) and \( P(2) = 3 \). If \( P(n) \) is a perfect square for all positive integers \( n \), find the smallest possible value of \( P(3) \).
**label:** `4`

### 26. (p̂=0.556, r_unc=-)
**Q:** In a certain kingdom, there are 100 cities, each connected by exactly 10 roads. No two cities are directly connected by more than one road, and no city is connected to itself. A traveler wants to visit as many cities as possible without traveling along any road more than once. What is the maximum number of cities the traveler can visit?
**label:** `100`

### 27. (p̂=0.333, r_unc=-)
**Q:** In a game, two players take turns placing tokens on a 4x4 grid. Each token covers exactly one cell, and no two tokens can overlap. Player 1 goes first, and the goal for each player is to create a path of their tokens from the top-left corner to the bottom-right corner, without stepping on any cell of the opponent's path. If Player 1 has a winning strategy, what is the minimum number of tokens Player 1 must place on the board to guarantee a win, assuming both players play optimally?
**label:** `5`

### 28. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the sum of the first \( n \) positive integers is a perfect square. That is, determine \( n \) so that
\[ 1 + 2 + 3 + \cdots + n = k^2 \]
for some integer \( k \).
**label:** `8`

### 29. (p̂=0.667, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation \( x^2 + y^2 = nz^2 \) has a solution in positive integers \( x, y, z \) with \( \gcd(x, y, z) = 1 \). Prove your answer.
**label:** `1`

### 30. (p̂=0.444, r_unc=-)
**Q:** Let $S$ be the set of all positive integers that can be represented as the sum of distinct powers of 3. Find the smallest positive integer $n$ such that the set $\{1, 2, \ldots, n\}$ is not a subset of $S$.
**label:** `2`

### 31. (p̂=0.778, r_unc=-)
**Q:** In a tournament with \( n \) teams, each team plays every other team exactly once. A win earns a team 3 points, a draw earns 1 point, and a loss earns 0 points. At the end of the tournament, the total points accumulated by all teams are 120. Find the smallest possible value of \( n \).
**label:** `10`

### 32. (p̂=0.556, r_unc=-)
**Q:** In a convex quadrilateral ABCD, the lengths of the sides are AB = 7, BC = 10, CD = 15, and DA = 20. The diagonals AC and BD intersect at point E. If the area of triangle ABE is 14 square units, what is the area of triangle CDE?
**label:** `30`

### 33. (p̂=0.667, r_unc=-)
**Q:** In a small village, there are 10 houses in a row. The owner of each house decides to paint their house either red, blue, or green. However, no two adjacent houses can be painted the same color. If the owner of the first house chooses to paint it red, how many different ways can the rest of the houses be painted to satisfy the condition?
**label:** `512`

### 34. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation
\[
\sqrt{n^2 + n + 1} + \sqrt{n^2 - n + 1} = \sqrt{n^2 + 2n + 2}
\]
has an integer solution for \( n \).
**label:** `1`

### 35. (p̂=0.444, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(1) = 2 \) and \( P(2) = 3 \). Suppose that for some integer \( n \), the polynomial \( Q(x) = P(x) + nx \) has all integer roots. Determine the smallest positive value of \( n \) for which this is possible.
**label:** `1`

### 36. (p̂=0.667, r_unc=-)
**Q:** What is the sum of all positive integers less than 1000 that are divisible by 5 but not by 15?
**label:** `66335`

### 37. (p̂=0.333, r_unc=-)
**Q:** Let \( ABCD \) be a cyclic quadrilateral with \( AB = 5 \), \( BC = 6 \), \( CD = 7 \), and \( DA = 8 \). Let \( \theta \) be the measure of the angle \( \angle ABC \). Find the value of \( \theta \) in degrees if \( \cos(\theta) = \frac{m}{n} \) where \( m \) and \( n \) are coprime positive integers. What is \( m + n \)?

(Note: A cyclic quadrilateral is a four-sided figure inscribed in a circle.)
**label:** `56`

### 38. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation \( x^2 + y^2 = nz^2 \) has no solutions in positive integers \( x, y, \) and \( z \).
**label:** `3`

### 39. (p̂=0.444, r_unc=-)
**Q:** Find the number of positive integers \( n \) such that the equation
\[ n^2 + 1 = 2^a + 2^b + 2^c \]
has at least two distinct solutions in non-negative integers \( a, b, \) and \( c \).
**label:** `1`

### 40. (p̂=0.778, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation
\[ x^2 + ny^2 = z^2 \]
has integer solutions \((x, y, z)\) with \( \gcd(x, y, z) = 1 \).
**label:** `1`


## iter 3  (N=4376, showing 40)

### 1. (p̂=0.444, r_unc=-)
**Q:** Let \( S \) be the set of all positive integers that can be expressed as the sum of two distinct positive integers, each of which has at least two distinct prime factors. What is the smallest positive integer \( n \) such that \( n \) cannot be expressed as the sum of two distinct elements of \( S \)?
**label:** `1`

### 2. (p̂=0.778, r_unc=-)
**Q:** Find the number of ordered pairs of integers $(x, y)$ such that $x^2 + y^2 = 2023$ and $x + y$ is a multiple of 5.
**label:** `0`

### 3. (p̂=0.333, r_unc=-)
**Q:** Find the number of ordered pairs \((a, b)\) of positive integers such that the equation \(a^b + b^a = 1999\) holds.
**label:** `2`

### 4. (p̂=0.667, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 - nx^2 + (n-1)x - 1 \) has at least one real root and the sum of the squares of its roots is an integer.
**label:** `1`

### 5. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 - nx^2 + (n-1)x - 1 \) has three distinct integer roots.
**label:** `3`

### 6. (p̂=0.333, r_unc=-)
**Q:** Let \( P(x) \) be a polynomial with integer coefficients such that \( P(0) = 1 \) and \( P(1) = 5 \). If \( P(x) \) has exactly three distinct integer roots, what is the sum of the absolute values of these roots?
**label:** `4`

### 7. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^4 - nx^3 + (n-1)x^2 + (n-2)x - 2 \) has four distinct positive integer roots.
**label:** `10`

### 8. (p̂=0.444, r_unc=-)
**Q:** In a triangle \(ABC\), the sides \(AB\), \(BC\), and \(CA\) are of lengths \(a\), \(b\), and \(c\) respectively. Let \(P\) be a point inside the triangle such that the perpendicular distances from \(P\) to the sides \(AB\), \(BC\), and \(CA\) are \(d_1\), \(d_2\), and \(d_3\) respectively. If the area of the triangle \(ABC\) is \(S\), find the maximum possible value of the product \(d_1 d_2 d_3\) in terms of \(a\), \(b\), and \(c\).
**label:** `\frac{8S^3}{27abc}`

### 9. (p̂=0.556, r_unc=-)
**Q:** In a finite grid of squares, each square is colored red or blue. A square is considered *favorable* if it has an odd number of red neighbors (including itself). If the total number of favorable squares is 2024, what is the minimum number of red squares that must be in the grid?
**label:** `2024`

### 10. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^2 + nx + 2024 \) has integer roots and the sum of its roots is a perfect square.
**label:** `2025`

### 11. (p̂=0.667, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 + ax^2 + bx + c \) has three distinct positive integer roots and the following conditions are satisfied:
1. The sum of the squares of the roots is equal to \( n \).
2. The product of the roots is a perfect square.
**label:** `49`

### 12. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^n + x^{n-1} + \cdots + x + 1 \) is divisible by \( x^2 - 2x + 2 \).
**label:** `7`

### 13. (p̂=0.444, r_unc=-)
**Q:** In a small town, there are 10 houses, each with a unique number of residents ranging from 1 to 10. The town council decides to organize a community event where every resident must attend at least one of the three events: a picnic, a concert, or a movie night. Each event can accommodate a maximum of 15 people, and no resident can attend more than one event. Given that the total number of attendees for all three events is exactly 20, determine the minimum number of houses where a resident must attend at least two events, and prove your reasoning.
**label:** `1`

### 14. (p̂=0.444, r_unc=-)
**Q:** Find all positive integers \( n \) such that the polynomial \( P(x) = x^n + x^{n-1} + \cdots + x + 1 \) is divisible by the polynomial \( Q(x) = x^2 - x - 1 \).
**label:** `6`

### 15. (p̂=0.667, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that there exists a polynomial \( P(x) \) of degree \( n \) with integer coefficients where \( P(1) = 2023 \), and for every prime \( p \), there exists an integer \( k \) such that \( P(k) \equiv 0 \pmod{p} \).
**label:** `1`

### 16. (p̂=0.778, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 - nx^2 + (n-1)x - 1 \) has three distinct real roots.
**label:** `4`

### 17. (p̂=0.778, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^n + ax^{n-1} + bx^{n-2} + \cdots + kx + l \) with integer coefficients has at least one non-real root, and \( a, b, \ldots, k, l \) are consecutive integers, starting from \( a = 1 \).
**label:** `2`

### 18. (p̂=0.556, r_unc=-)
**Q:** Find the number of ordered pairs of integers \((x, y)\) such that the equation \(x^2 + y^2 = x^3\) holds and \(x\) and \(y\) are both positive.
**label:** `1`

### 19. (p̂=0.667, r_unc=-)
**Q:** What is the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 - nx^2 + (n+2)x - 3 \) has three distinct integer roots?
**label:** `3`

### 20. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation \( x^2 + y^2 + z^2 = n \cdot xyz \) has a solution in positive integers \( x, y, z \) where \( x, y, \) and \( z \) are all distinct.
**label:** `3`

### 21. (p̂=0.778, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that \( n! \) (the factorial of \( n \)) ends with exactly 24 zeros when written in base 10.
**label:** `100`

### 22. (p̂=0.556, r_unc=-)
**Q:** Let \( S \) be a set of \( 2n \) distinct positive integers. Prove that there exists a subset \( T \) of \( S \) such that the sum of the elements in \( T \) is divisible by \( 2n \). Furthermore, if no such subset exists for any \( 2n \), determine the largest possible value of \( n \) for which this is impossible.
**label:** `1`

### 23. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^n + ax^{n-1} + bx^{n-2} + \cdots + cx + d \) with integer coefficients \( a, b, \ldots, d \) satisfies the condition that for all real numbers \( x \) and \( y \), the expression \( P(x+y) \) is divisible by \( P(x) \) and \( P(y) \).
**label:** `2`

### 24. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^4 - 2x^3 + ax^2 + bx + c \) has four distinct real roots, where \( a, b, \) and \( c \) are integers, and the product of the roots taken two at a time is equal to the sum of the roots taken one at a time.
**label:** `4`

### 25. (p̂=0.778, r_unc=-)
**Q:** Find all positive integers \( n \) such that the sum of the squares of the first \( n \) positive integers is equal to the square of a positive integer.
**label:** `1`

### 26. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that \( n! \) ends with exactly 12 zeros and \( n+1 \) is a prime number.
**label:** `52`

### 27. (p̂=0.667, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the sum of the first \( n \) positive integers, when divided by \( n \), is a perfect square.
**label:** `1`

### 28. (p̂=0.556, r_unc=-)
**Q:** Find the number of positive integers \( n \) such that \( n^2 + 14n + 48 \) is a perfect square.
**label:** `0`

### 29. (p̂=0.778, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that there exists an integer \( k \) with the property that \( n^2 + k \) is divisible by \( 2n + k \) and \( n^2 - k \) is divisible by \( 2n - k \). What is the value of \( n \)?
**label:** `2`

### 30. (p̂=0.444, r_unc=-)
**Q:** In a special deck of cards, each card is labeled with a unique integer from 1 to 100. Alice draws two cards at random and sums their labels. Bob then draws two cards at random from the remaining 98 cards and sums their labels. If the probability that Alice's sum is exactly double Bob's sum is $p$, find the value of $100p$ rounded to the nearest integer.
**label:** `1`

### 31. (p̂=0.667, r_unc=-)
**Q:** Find the number of positive integers \( n \) such that the polynomial \( P(x) = x^3 - 3x^2 + nx + 2024 \) has at least one real root that is an integer.
**label:** `0`

### 32. (p̂=0.778, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^n + x^{n-1} + \cdots + x + 1 \) can be written as the product of two non-constant polynomials with integer coefficients.
**label:** `3`

### 33. (p̂=0.333, r_unc=-)
**Q:** Find all positive integers \( n \) such that the polynomial \( P(x) = x^3 - nx^2 + (n-1)x - 1 \) has integer roots.
**label:** `3`

### 34. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation \( x^2 + y^2 + z^2 = n(x + y + z) \) has at least three distinct non-negative integer solutions where \( x, y, \) and \( z \) are not all zero.
**label:** `1`

### 35. (p̂=0.556, r_unc=-)
**Q:** What is the smallest positive integer \( n \) such that the polynomial \( P(x) = x^n + x^{n-1} + \cdots + x + 1 \) is divisible by the polynomial \( Q(x) = x^2 + ax + b \), where \( a \) and \( b \) are integers with \( |a| \leq 10 \) and \( |b| \leq 10 \)?
**label:** `2`

### 36. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 - nx^2 + (n-1)x - 1 \) has at least one integer root and the sum of the squares of its roots is 7.
**label:** `4`

### 37. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation
\[
x^2 + y^2 + z^2 = nxyz
\]
has no non-trivial integer solutions (i.e., solutions where \( x, y, z \) are not all zero and not all positive).
**label:** `3`

### 38. (p̂=0.778, r_unc=-)
**Q:** In a small town, there are 10 houses connected by a network of roads. Each house is directly connected to at least two other houses, but no three houses form a triangle (i.e., there is no set of three houses where each pair is directly connected). What is the maximum number of roads that can exist in this town?
**label:** `25`

### 39. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) that is divisible by the largest prime less than \( 247 \times 29 + 42 \) and satisfies the condition that \( n^2 \equiv 1 \mod n \). Determine the number of divisors of \( n \) and their sum.
**label:** `7200`

### 40. (p̂=0.444, r_unc=-)
**Q:** A sequence of positive integers is defined recursively by \(a_1 = 1\) and \(a_{n+1} = a_n^2 + a_n\) for \(n \geq 1\). Find the smallest positive integer \(k\) such that \(a_k\) has at least 2017 digits.
**label:** `12`


## iter 4  (N=5387, showing 40)

### 1. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 - 2x^2 + x + n \) has three distinct roots, all of which are integers, and the product of any two roots is not a prime number.
**label:** `2`

### 2. (p̂=0.778, r_unc=-)
**Q:** Find the smallest positive integer n such that the polynomial P(x) = x^n + ax^(n-1) + bx^(n-2) + ... + cz^2 + dx + e has the property that the sum of its coefficients is equal to the product of its constant term and the sum of its linear term coefficients.
**label:** `2`

### 3. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that there exist integers \( a \), \( b \), and \( c \) satisfying the equation
\[ n^2 + a^2 + b^2 = 2nab. \]
**label:** `1`

### 4. (p̂=0.667, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 + ax^2 + bx + c \) has roots that are all integers, and the polynomial \( Q(x) = x^3 + dx^2 + ex + f \) formed by interchanging the roots of \( P(x) \) with their squares also has integer roots. Determine the value of \( n \).
**label:** `1`

### 5. (p̂=0.667, r_unc=-)
**Q:** Find the smallest positive integer $n$ such that the polynomial $P(x) = x^3 - x^2 + nx + 2n$ has three distinct integer roots $a$, $b$, and $c$ with $a < b < c$ and $a + b + c > 0$.
**label:** `2`

### 6. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation \( x^2 + y^2 = nxy \) has no integer solutions where \( x \) and \( y \) are distinct positive integers. 

Prove that such an \( n \) exists and find its value.
**label:** `2`

### 7. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer $n$ for which there exists a real number $x$ such that $n \sin x + \frac{100}{\sin x} = n^2$.
**label:** `8`

### 8. (p̂=0.333, r_unc=-)
**Q:** Find all positive integers \( n \) such that \( n^2 + 3n + 2 \) divides \( n^3 + 2n + 6 \). Determine the sum of all such \( n \).
**label:** `0`

### 9. (p̂=0.667, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^n + ax^{n-1} + bx^{n-2} + \cdots + cx + d \) with integer coefficients satisfies the following conditions:
1. \( P(1) = 1994 \)
2. \( P(1994) = n \)
**label:** `1994`

### 10. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^4 + ax^3 + bx^2 + cx + d \) with integer coefficients and \( P(1) = P(2) = P(3) = P(4) = 0 \) has the form \( P(x) = (x - 1)(x - 2)(x - 3)(x - 4) \) and \( a + b + c + d = 2023 \).
**label:** `1`

### 11. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that \( n^2 + 15n + 56 \) is a perfect square.
**label:** `1`

### 12. (p̂=0.556, r_unc=-)
**Q:** Find all positive integers \( n \) such that \( n^2 + 3n + 2 \) is a perfect square.
**label:** `1`

### 13. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer $n$ such that the polynomial $P(x) = x^3 - nx^2 + (n-1)x - 1$ has at least one integer root and the sum of its roots is an integer multiple of its constant term.
**label:** `3`

### 14. (p̂=0.556, r_unc=-)
**Q:** What is the smallest positive integer \( n \) such that the equation
\[ x^2 + y^2 + z^2 = nxyzt \]
has no integer solutions where \( x, y, z, t \) are distinct integers greater than 1?
**label:** `1`

### 15. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 - nx^2 + (n-1)x - 1 \) has three distinct integer roots.
**label:** `3`

### 16. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that \( n^2 + 15n + 56 \) is a perfect square.
**label:** `1`

### 17. (p̂=0.556, r_unc=-)
**Q:** Find all positive integers \( n \) such that \( n^2 + 3n + 2 \) is a perfect square.
**label:** `1`

### 18. (p̂=0.778, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 + ax^2 + bx + c \) with integer coefficients has three distinct roots, one of which is a root of \( Q(x) = x^4 + dx^3 + ex^2 + fx + g \) with integer coefficients, and the polynomial \( Q(x) \) also has a rational root.
**label:** `1`

### 19. (p̂=0.750, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 + ax^2 + bx + c \) has three distinct integer roots, and the polynomial \( Q(x) = x^3 + (a-1)x^2 + (b-2)x + (c-3) \) also has three distinct integer roots. What is the value of \( n \)?
**label:** `6`

### 20. (p̂=0.778, r_unc=-)
**Q:** Find the smallest positive integer n such that n^2 + 15n + 56 is a perfect square, given that n is also a solution to the equation 2n^2 - 3n - 20 = 0.
**label:** `4`

### 21. (p̂=0.778, r_unc=-)
**Q:** Find the smallest positive integer $n$ such that the polynomial $P(x) = x^3 - nx^2 + (n-1)x - 1$ has three distinct integer roots, and the sum of the squares of these roots is a perfect square.
**label:** `3`

### 22. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation \( x^2 + y^2 = nxy \) has exactly three pairs of positive integer solutions \( (x, y) \).
**label:** `3`

### 23. (p̂=0.333, r_unc=-)
**Q:** Find the number of ordered pairs $(a, b)$ of integers such that the equation $x^2 + ax + b = 0$ has at least one root, where the root is also a non-integer root of the equation $y^2 + ay + b^2 = 0$.

[Hint: Use the discriminant of the quadratic equations involved to establish a relationship between $a$ and $b$ and apply modular arithmetic concepts.]
**label:** `2`

### 24. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation
\[
x^2 + ny^2 = (n+1)(x + y)
\]
has at least four distinct pairs of integer solutions \((x, y)\).
**label:** `1`

### 25. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that there exists a set \( S \) of \( n \) points in the plane satisfying the following conditions:
1. No three points in \( S \) are collinear.
2. For every point \( P \) in \( S \), the number of lines passing through \( P \) and containing at least two points of \( S \) is an even number.
**label:** `3`

### 26. (p̂=0.444, r_unc=-)
**Q:** Find all positive integers \( n \) for which the equation
\[ x^2 + ny^2 = (n+1)x \]
has exactly two distinct integer solutions \((x_1, y_1)\) and \((x_2, y_2)\).
**label:** `1`

### 27. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^4 - 2x^3 + 3x^2 - 4x + n \) has exactly two distinct positive real roots and both of these roots are irrational.
**label:** `2`

### 28. (p̂=0.556, r_unc=-)
**Q:** Find all positive integers \( n \) such that \( n^2 + 3n + 2 \) is a perfect square. Prove that no other values of \( n \) satisfy this condition.
**label:** `1`

### 29. (p̂=0.667, r_unc=-)
**Q:** Find the smallest positive integer n such that the polynomial p(x) = x^3 - nx^2 + (n-1)x - n has three distinct real roots, and the product of these roots is a perfect square.
**label:** `4`

### 30. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that \( n^2 + 15n + 56 \) is a perfect square.
**label:** `1`

### 31. (p̂=0.333, r_unc=-)
**Q:** Find all positive integers \( n \) for which \( n^2 + 3n + 2 \) is a perfect square.
**label:** `0`

### 32. (p̂=0.778, r_unc=-)
**Q:** Find all positive integers \( n \) such that \( n^2 + 3n + 2 \) is a perfect square.

(a) Verify that \( n = 2 \) is a solution.

(b) Prove that there are no other solutions.
**label:** `2`

### 33. (p̂=0.333, r_unc=-)
**Q:** Let \( S \) be a set of positive integers. We say that \( S \) is *prime-packed* if every integer in \( S \) can be uniquely represented as a product of distinct prime numbers. 

Define a function \( f(S) \) as follows: If \( S \) is prime-packed, then \( f(S) \) is the smallest positive integer that cannot be expressed as a product of any subset of the elements of \( S \). If \( S \) is not prime-packed, then \( f(S) \) is the smallest positive integer that can be expressed in more than one way as a product of a subset of the elements of \( S \).

For example, if \( S = \{2, 3, 5\} \), which is prime-packed, then \( f(S) = 6 \) since \( 6 \) cannot be expressed as a product of any subset of \( S \).

Now, consider the set \( T = \{2, 3, 5, 7, 11, 13\} \). Find \( f(T) \).
**label:** `4`

### 34. (p̂=0.778, r_unc=-)
**Q:** Find all positive integers \( n \) such that there exists a positive integer \( k \) for which the polynomial \( P(x) = x^n - kx^{n-1} + kx - 1 \) has exactly \( n \) distinct integer roots.
**label:** `1`

### 35. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation \( x^2 + y^2 = nxy \) has at least two distinct positive integer solutions for \( x \) and \( y \), and furthermore, the sum \( x + y \) is also divisible by \( n \).
**label:** `3`

### 36. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 - x^2 - (n-1)x + n \) has three distinct real roots, all of which are also integers.
**label:** `3`

### 37. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation \( x^2 + y^2 = nz^2 \) has at least one solution in positive integers \( x, y, z \) with \( \gcd(x, y, z) = 1 \), and \( x, y, z \) are not all multiples of the same prime number.
**label:** `1`

### 38. (p̂=0.444, r_unc=-)
**Q:** Find all positive integers \( n \) such that the equation \( n^2 + n + 1 = m^3 \) has an integer solution \( m \). Prove that your answer is exhaustive.
**label:** `0`

### 39. (p̂=0.333, r_unc=-)
**Q:** Find all positive integers $n$ such that both $n$ and $n^2+17$ are powers of primes.
**label:** `0`

### 40. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 - nx^2 + (n-1)x - 1 \) has at least one integer root and all its roots are distinct integers.
**label:** `3`


## iter 5  (N=6270, showing 40)

### 1. (p̂=0.778, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation \( \frac{1}{x} + \frac{1}{y} + \frac{1}{z} = \frac{1}{n} \) has at least three distinct positive integer solutions \((x, y, z)\).
**label:** `6`

### 2. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^4 + ax^3 + bx^2 + cx + d \) with integer coefficients has four distinct integer roots and \( P(n) = 10n \).
**label:** `2`

### 3. (p̂=0.778, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^4 - nx^3 + (n-1)x^2 - nx + 1 \) has four distinct integer roots.
**label:** `4`

### 4. (p̂=0.444, r_unc=-)
**Q:** Let \( S \) be the set of all ordered pairs \((a, b)\) of integers satisfying \( a^2 + b^2 \equiv 0 \pmod{p} \), where \( p \) is a prime number. If \( p \equiv 1 \pmod{4} \), prove that the number of such pairs is divisible by 4, and determine whether this statement holds for \( p \equiv 3 \pmod{4} \).
**label:** `4`

### 5. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^4 + ax^3 + bx^2 + cx + d \) has four distinct real roots, the sum of any two roots is not equal to the square of the third root, and the product of the roots taken two at a time is equal to \( n^2 \).
**label:** `1`

### 6. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation
\[
x^2 + ny^2 = p
\]
has no integer solutions for any prime number \( p \) with \( 100 < p < 200 \).
**label:** `2`

### 7. (p̂=0.778, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation
\[ x^2 + ny^2 = z^2 \]
has infinitely many solutions in integers \( x, y, z \) with \( x, y, z > 0 \) and \( \gcd(x, y, z) = 1 \).
**label:** `1`

### 8. (p̂=0.778, r_unc=-)
**Q:** What is the smallest positive integer n such that n^2 + 15n + 56 is a perfect square?
**label:** `1`

### 9. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer $n$ such that the equation
\[ x^2 + ny^2 = n^2x \]
has at least two distinct integer solutions $(x_1, y_1)$ and $(x_2, y_2)$ with $x_1 \neq x_2$ and $y_1 \neq y_2$.
**label:** `2`

### 10. (p̂=0.556, r_unc=-)
**Q:** Find the number of ordered pairs $(a, b)$ of integers such that the equation $x^2 + ax + b = 0$ has two distinct integer roots, and the equation $x^2 + bx + a = 0$ also has two distinct integer roots. Additionally, determine the smallest possible value of $a + b$.
**label:** `-1`

### 11. (p̂=0.667, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation
\[
x^2 + ny^2 = (x + y)(x + y + 1)
\]
has at least three distinct pairs of integer solutions \((x, y)\).
**label:** `2`

### 12. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^4 + ax^3 + bx^2 + cx + d \) has four distinct integer roots, and the sum of the squares of these roots is equal to \( n \).
**label:** `10`

### 13. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer $n$ such that the equation $x^2 + ny^2 = n^2$ has exactly three distinct integer solutions $(x,y)$.
**label:** `3`

### 14. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation \( x^2 + y^2 = nz^2 \) has at least three distinct pairs of integers \( (x, y) \) for which \( x + y \) is divisible by \( n \).
**label:** `2`

### 15. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^4 + ax^3 + bx^2 + cx + d \) with integer coefficients has exactly two real roots, both of which are integers, and the other two roots are complex conjugates with the condition that the imaginary part of each complex root is a prime number.
**label:** `2`

### 16. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that \( n^2 + 15n + 56 \) is a perfect square. Additionally, determine the smallest integer \( m \) such that \( n^2 + 15n + 56 = m^2 \) and \( m + n \) is also a perfect square.
**label:** `1`

### 17. (p̂=0.667, r_unc=-)
**Q:** What is the probability that a player can force a win in a game where they construct a graph with 6 vertices and 9 edges, ensuring that the resulting graph contains an Eulerian cycle, but must also prevent the opponent from creating a Hamiltonian path during their turn? Express your answer as a simplified fraction.
**label:** `1`

### 18. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^4 + ax^3 + bx^2 + cx + d \) has four distinct integer roots, where \( a, b, c, \) and \( d \) are all positive integers, and the sum of the reciprocals of its roots is an integer.
**label:** `4`

### 19. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^4 - nx^3 + (n-1)x^2 - nx + 1 \) has four distinct positive real roots.
**label:** `5`

### 20. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial equation
\[ x^3 + ax^2 + bx + c = 0 \]
has roots \( \alpha, \beta, \gamma \) that satisfy the condition:
\[ \alpha^3 + \beta^3 + \gamma^3 - 3\alpha\beta\gamma = 2023. \]
**label:** `1`

### 21. (p̂=0.667, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation
\[ \frac{1}{x} + \frac{1}{y} = \frac{1}{n} \]
has at least three distinct pairs of positive integer solutions \((x, y)\).
**label:** `6`

### 22. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^4 + ax^3 + bx^2 + cx + d \) has four distinct integer roots and the product of any two roots is not equal to the sum of the other two roots.
**label:** `1`

### 23. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that there exist integers \( a, b, c, d \) with the following properties:
1. \( a^2 + b^2 + c^2 + d^2 = n \)
2. \( a + b + c + d = 0 \)
3. The product \( abcd \) is divisible by 4, but not by 8.
**label:** `4`

### 24. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer n such that the polynomial f(x) = x^3 + ax^2 + bx + c has three distinct integer roots and the sum of their squares is equal to n.
**label:** `14`

### 25. (p̂=0.667, r_unc=-)
**Q:** Find the smallest positive integer $n$ such that the polynomial $P(x) = x^4 - 2x^3 + (n-1)x^2 - nx + n$ has at least two distinct integer roots.
**label:** `2`

### 26. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^4 - nx^3 + (n-1)x^2 - 3x + 2 \) has four distinct integer roots.
**label:** `6`

### 27. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^4 - nx^3 + (n-1)x^2 - nx + 1 \) has four distinct real roots, each of which is a root of unity.
**label:** `3`

### 28. (p̂=0.667, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 + ax^2 + bx + c \) has three distinct integer roots, and the equation \( P(x) = n \) has exactly three distinct integer solutions.
**label:** `0`

### 29. (p̂=0.556, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that \( n^2 + 15n + 56 \) is a perfect square, and determine the next two consecutive integers for which the same quadratic expression is also a perfect square.
**label:** `1`

### 30. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^n + 10x^{n-1} + 38x^{n-2} + \cdots + 10x + 1 \) is divisible by another polynomial \( Q(x) = x^2 + ax + b \) where \( a \) and \( b \) are integers with \( a + b = -1 \).
**label:** `4`

### 31. (p̂=0.778, r_unc=-)
**Q:** Find the smallest positive integer n such that the polynomial P(x) = x^4 + ax^3 + bx^2 + cx + d has exactly one real root and three non-real complex roots with all coefficients a, b, c, and d being integers, and the product of the non-real complex roots is an integer.
**label:** `1`

### 32. (p̂=0.667, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^4 - nx^3 + (n+2)x^2 - 3nx + 4 \) has four distinct integer roots.
**label:** `6`

### 33. (p̂=0.778, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^4 + ax^3 + bx^2 + cx + d \) has four distinct real roots, where \( a, b, c, \) and \( d \) are integers, and the roots form an arithmetic progression. What is the sum of the absolute values of all such \( n \)?
**label:** `1`

### 34. (p̂=0.667, r_unc=-)
**Q:** In the complex plane, the points $A, B, C$ correspond to the complex numbers $z_1, z_2, z_3$ respectively. Given that $|z_1| = |z_2| = |z_3| = 1$ and $z_1^2 + z_2^2 + z_3^2 = 0$, find the maximum possible value of
$$
\text{Re}(z_1^3 + z_2^3 + z_3^3).
$$
**label:** `3`

### 35. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the equation \( \frac{1}{x} + \frac{1}{y} + \frac{1}{z} = \frac{1}{n} \) has exactly six distinct positive integer solutions \((x, y, z)\) where \( x, y, z \leq n \).
**label:** `12`

### 36. (p̂=0.778, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 + ax^2 + bx + c \) with integer coefficients has three distinct real roots, and the product of any two of its roots is a perfect square.
**label:** `1`

### 37. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 + ax^2 + bx + c \) has three distinct integer roots, and the sum of the cubes of these roots is equal to the square of the sum of the roots.
**label:** `1`

### 38. (p̂=0.333, r_unc=-)
**Q:** Find the smallest positive integer $n$ such that $n^2 + 14n + 48$ is a perfect square, and $n + 14$ is also a prime number.
**label:** `3`

### 39. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^n + ax^{n-1} + bx^{n-2} + cx^{n-3} + d \) has the property that the sum of its roots taken four at a time is equal to zero, given that the polynomial has exactly one real root and the remaining roots are purely imaginary. Also, ensure that the polynomial satisfies \( P(1) = 2 \).
**label:** `4`

### 40. (p̂=0.444, r_unc=-)
**Q:** Find the smallest positive integer \( n \) such that the polynomial \( P(x) = x^3 + ax^2 + bx + c \) has three distinct integer roots, and the polynomial \( Q(x) = x^4 + dx^3 + ex^2 + fx + g \) has roots that are exactly the squares of the roots of \( P(x) \), where \( a, b, c, d, e, f, \) and \( g \) are integers.
**label:** `2`
