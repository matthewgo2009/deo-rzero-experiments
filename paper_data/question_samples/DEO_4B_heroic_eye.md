# DEO_4B_heroic_eye — question samples


_DEO MCMC walk, Qwen3-4B (full stack: CD n=12 + strong-β + KL-prev, 2000-q pool)_

Per iteration: total pool size N, then an evenly-spaced sample of 40 questions.
Fields — `q`: question text; `label`: pseudo-label/answer used for training; `gt`: challenger-proposed answer (DEO only); `phat`: self-consistency p̂ = modal-count/m (R-Zero `score` is the same quantity); `runc`: uncertainty reward 1−2|p̂−0.5| (DEO only).


## iter 1  (N=2000, showing 40)

### 1. (p̂=0.250, r_unc=0.500)
**Q:** Given a $60^\circ$ wedge of size $N$. 
\begin{figure}
\centering
\input{corners.tex}
\caption{A $60^\circ$ wedge divided into four  $60^\circ$ sectors. }
\end{figure} 
A smaller square $PQRS$ is placed inside the shaded region of the wedge such that each vertex  of the smaller square lies on a different side of the small square's opposite wedge as pictured.  How many such smaller squares are there whose area is equal to the area of the smallest wedge?
**label:** `4` | gt(challenger)=`4`

### 2. (p̂=0.583, r_unc=0.833)
**Q:** A cube of side length \( s \) is circumscribed by a sphere of radius \( R \). Find an expression for the ratio of the surface area of the cube to the surface area of the sphere in terms of \( s \) and \( R \).
**label:** `\frac{2}{\pi}` | gt(challenger)=`\frac{2}{\pi}`

### 3. (p̂=0.833, r_unc=0.333)
**Q:** Find the limit of the function \( f(x) = \frac{x^3 - 3x^2 + 2x}{x^2 - 4} \) as \( x \) approaches 2.
**label:** `\frac{1}{2}` | gt(challenger)=`\frac{1}{2}`

### 4. (p̂=0.250, r_unc=0.500)
**Q:** Given quadrilateral \(ABCD\) with \(AB = 8\), \(BC = 12\), \(CD = 10\), and \(DA = 16\), find the sum of the perpendicular distances from a point \(P\) inside the quadrilateral to its four sides.
**label:** `24` | gt(challenger)=`24`

### 5. (p̂=0.167, r_unc=0.333)
**Q:** In \( \triangle ABC \), the incircle touches \( BC, CA, \) and \( AB \) at \( D, E, \) and \( F \) respectively. Given that \( \triangle ABC \) is a right triangle with \( \angle ACB = 90^\circ \) and \( AC = 12 \), find the sum of all possible integer values of \( BC \) if the radius of the incircle is a prime number.
**label:** `37` | gt(challenger)=`37`

### 6. (p̂=0.583, r_unc=0.833)
**Q:** **How many distinct positive integers \( n \) are there such that \( n \) is the smallest positive integer for which \( n^2 + (n + 1)^2 \) is divisible by \( 4n \) and \( n \) is also a prime number?
**label:** `0` | gt(challenger)=`0`

### 7. (p̂=0.250, r_unc=0.500)
**Q:** Find all complex numbers $z$ such that $z^2 + z + 1 = 0$.
**label:** `\frac{-1 + i\sqrt{3}}{2}, \frac{-1 - i\sqrt{3}}{2}` | gt(challenger)=`e^{i\pi/3}, e^{i\pi/3 + i\pi}`

### 8. (p̂=0.167, r_unc=0.333)
**Q:** In a round table, 6 people are to be seated. If two specific people, Alex and Ben, must cover an angle of exactly 180° while looking at each other, how many degrees must the angle subtended by the table be to accommodate this condition? \boxed{
**label:** `360` | gt(challenger)=`360`

### 9. (p̂=0.250, r_unc=0.500)
**Q:** Alice has 5 red balls and 3 blue balls. She randomly picks balls one by one without replacement until she picks a blue ball. Let \( E \) be the expected number of red balls she picks before picking the first blue ball. Additionally, calculate the probability that she picks exactly 2 red balls before picking the first blue ball. Calculate \( E \) and this probability.
**label:** `\frac{5}{28}` | gt(challenger)=`\frac{5}{28}`

### 10. (p̂=0.250, r_unc=0.500)
**Q:** Translation: Determine the smallest positive integer $n$ for which $g^n(1) < 10$, and exhibiting higher partial deficiency to $\pm 1$ in $g$.

Given the transformation function $g(x) = x^2 + 3x + 2$, find the smallest positive integer $n$ such that $g^n(1) < 10$, with the additional constraint that the final value is closer to $0$ (which in this case would be equivalent to being closer to $1$ due to the simplicity of the problem).
**label:** `1` | gt(challenger)=`1`

### 11. (p̂=0.500, r_unc=1.000)
**Q:** A regular tetrahedron has edges of length 6 units. A smaller, similar tetrahedron is inscribed within it, touching all four faces at the midpoints of each edge. What is the ratio of the volume of the inscribed tetrahedron to the volume of the original tetrahedron?
**label:** `\frac{1}{8}` | gt(challenger)=`1:8`

### 12. (p̂=0.500, r_unc=1.000)
**Q:** Find all integer solutions \(x\) and \(y\) to the equation \(x^2 - 4y^2 = 15\).
**label:** `(none)` | gt(challenger)=`final_answer`

### 13. (p̂=0.167, r_unc=0.333)
**Q:** Given four distinct points \( A, B, C, \) and \( D \) in the plane, no three of which are collinear, let their coordinates be \( A(1,1), B(4,2), C(5,5), \) and \( D(3,4) \). Construct the parabolas \( y = a(x - x_A)^2 + y_A \) for \( A \), \( y = a(x - x_B)^2 + y_B \) for \( B \), \( y = a(x - x_C)^2 + y_C \) for \( C \), and \( y = a(x - x_D)^2 + y_D \) for \( D \), where \( x_A, x_B, x_C, \) and \( x_D \) are the \( x \)-coordinates of points \( A, B, C, \) and \( D \), respectively. 

Let \( P \) be the point of intersection of the parabolas corresponding to \( A \) and \( B \), and \( Q \) be the point of intersection of the parabolas corresponding to \( C \) and \( D \). Compute the distance \( PQ \).
**label:** `\sqrt{2}` | gt(challenger)=`\sqrt{17}`

### 14. (p̂=0.083, r_unc=0.167)
**Q:** You want to enclose a rectangular garden with the least amount of material possible. In addition to the following conditions:

the area of the garden must be 100 square meters;
the length of the garden must be twice as long as its width.

What will be the dimensions of the garden?
**label:** `` | gt(challenger)=`(10,20)`

### 15. (p̂=0.250, r_unc=0.500)
**Q:** For each positive integer \( n \), let \( S(n) \) be the sum of the digits of \( n \). Find the smallest positive integer \( n \) such that \( n + S(n) + S(S(n)) = 2021 \).
**label:** `2018` | gt(challenger)=`1983`

### 16. (p̂=0.167, r_unc=0.333)
**Q:** Consider a rectangle with length \( L \) and width \( W \). You are given that the area of this rectangle is \( 100 \) square units. What are the dimensions \( L \) and \( W \) of the corresp
orted rectangle with area \( \frac{100}{\pi} \) square units?
**label:** ``

### 17. (p̂=0.333, r_unc=0.667)
**Q:** Given real numbers \(a\), \(b\), \(c\), \(d\), \(e\) that satisfy the inequality \(\frac{a}{b} > \frac{c}{d} > \frac{e}{a}\), find additional conditions such that \(abcd > 0\) and \(bde < 0\). Determine if the additional conditions allow for more than two possible arrangements of the midpoints of two skew segments lying in a pair of parallel planes. If so, how many different arrangements are possible? If not, prove why.
**label:** `2` | gt(challenger)=`2`

### 18. (p̂=0.250, r_unc=0.500)
**Q:** Let \(S\) be the set of all distinct ordered quadruples \((a, b, c, d)\) of positive integers such that
\[a^2 - 2b = 3c^2\quad\text{and}\quad b^2 - 2a = 3d^2,\]
and the product \(abcd\) is a perfect square. Find the expected value of \(|a| + |b| + |c| + |d|\) over all quadruples in \(S\).
**label:** `8` | gt(challenger)=`8`

### 19. (p̂=0.417, r_unc=0.833)
**Q:** In a regular tetrahedron with edge length \(a\), a smaller tetrahedron is inscribed such that its vertices touch the midpoints of the original tetrahedron's edges. Calculate the ratio of the volume of the inscribed tetrahedron to the original tetrahedron.
**label:** `\frac{1}{8}` | gt(challenger)=`\frac{1}{8}`

### 20. (p̂=0.417, r_unc=0.833)
**Q:** A solid rectangular prism with dimensions \( x \) units by \( y \) units by \( z \) units and a total volume of \( xyz \) cubic units is painted on all its faces. When the prism is cut into smaller identical cubes with edge length 1 unit, how many cubes have at least one face painted?
**label:** `xyz - (x-2)(y-2)(z-2)` | gt(challenger)=`xyz - (x-2)(y-2)(z-2)`

### 21. (p̂=0.083, r_unc=0.167)
**Q:** Start with a 4x4 grid where each cell is uniquely assigned a distinct integer from 1 to 16. What is the expected value of the sum of the products of each row and column in this 4x4 grid?
**label:** `0` | gt(challenger)=`0`

### 22. (p̂=0.333, r_unc=0.667)
**Q:** In a game where 15 European member states start, one is randomly chosen to be expelled each day until only three remain. If we consider the total number of expulsions over the course of a year, what is the expected number of expulsions in 2021?
**label:** `12` | gt(challenger)=`12`

### 23. (p̂=0.500, r_unc=1.000)
**Q:** Determine the limit of the function \(f(x, y) = \frac{xy^2 - 4xy + 4x}{x^2 - 4}\) as \((x, y)\) approaches \((2, 2)\) via two different paths: \(x = 2\) and \(y = 2\). State if the limits along these paths are equal, and if they coincide with the value of the function at \((2, 2)\).
**label:** `0` | gt(challenger)=`0`

### 24. (p̂=0.417, r_unc=0.833)
**Q:** Let \( x \) and \( y \) be positive integers such that \( xy = 216 \). If their least common multiple (LCM) is also \( 216 \), find the smallest possible value of \( x + y \).
**label:** `35` | gt(challenger)=`35`

### 25. (p̂=0.333, r_unc=0.667)
**Q:** Find the number of positive integers less than 1000 that have more than 2 interchangeable pairs of digits.
**label:** `9` | gt(challenger)=`9`

### 26. (p̂=0.417, r_unc=0.833)
**Q:** Find the smallest positive integer n such that 7n ≡ 3 (mod 20).
**label:** `9` | gt(challenger)=`9`

### 27. (p̂=0.250, r_unc=0.500)
**Q:** There are now 7 red marbles, 3 blue marbles, and 4 green marbles in a bag. You draw two marbles twice in a row, record the number of red marbles drawn, and return the marbles to the bag only if the first marble drawn is blue. Calculate the expected number of red marbles drawn.
**label:** `1` | gt(challenger)=`1`

### 28. (p̂=0.833, r_unc=0.333)
**Q:** What is the limit of the function \( f(x) = \frac{x^2 - 4}{x - 2} \) as \( x \) approaches 2? Find the derivative of \( f(x) \) at \( x = 3 \).
**label:** `1` | gt(challenger)=`2, 1`

### 29. (p̂=0.417, r_unc=0.833)
**Q:** Consider all pairs of integers \( (n, m) \) such that \( n^2 - m^2 = k \) for some positive integer \( k \). Find the smallest positive integer \( n \) such that there exists an integer \( m \) satisfying \( n! \equiv 1 \pmod{5} \) and \( n^2 - m^2 = k \).\boxed{4}
**label:** `4` | gt(challenger)=`4`

### 30. (p̂=0.250, r_unc=0.500)
**Q:** In triangle \(ABC\), the angle bisector of \(\angle A\) intersects the circumcircle of \(\triangle ABC\) at points \(D\) and \(E\). Let \(F\) be the midpoint of \(BC\), and let the circle with diameter \(AF\) intersect the circumcircle of \(\triangle ABC\) at \(G\) (other than \(A\)). If \(AG = 8 + ABC\) and \(BG = 5\), find the length of \(FG\).
**label:** `3` | gt(challenger)=`3`

### 31. (p̂=0.167, r_unc=0.333)
**Q:** What is the volume of a tetrahedron with dimensions a, 8 units, and 10 units? Determine the maximum possible volume if variable a ranges from 4 units to 12 units.
**label:** `` | gt(challenger)=`1280`

### 32. (p̂=0.333, r_unc=0.667)
**Q:** In vector algebra, vectors 𢄃, 𢈍, and 𢄏 have magnitudes of 5, 12, and 13, respectively, and make up a triangle in three-dimensional space. What is the smallest value of n, for which there exists n vectors each obeying the same vector rules as positional seen in the framework above, having a coordinate in one of the three-dimensional units that way the sum of his coordinates envelope null vector?
**label:** `6` | gt(challenger)=`6`

### 33. (p̂=0.583, r_unc=0.833)
**Q:** In the coordinate plane, points $A(0,0)$, $B(8,0)$, and $C(0,6)$ form triangle $ABC$. A point $P(x,y)$ is selected on the perpendicular bisector of segment $BC$. Find the value of $x+y$ if the area of triangle $ABP$ is 24 square units.
**label:** `\frac{49}{4}` | gt(challenger)=`4`

### 34. (p̂=0.500, r_unc=1.000)
**Q:** Given that \( n \) is a positive integer satisfying \( n^a \equiv 1 \pmod{15} \) for some prime factor \( p \) of \( 15 \) raised to a positive power, find the smallest possible value of \( n \) in terms of \( a \) whenever \( p = 3 \).
**label:** `1` | gt(challenger)=`1`

### 35. (p̂=0.083, r_unc=0.167)
**Q:** In a variant of Pascal's triangle, instead of starting with 1s at the endpoints, each number is the sum of twice the number above it to the left plus the number diagonally above it to the right. What is the value in the 8th row and 4th column of this modified Pascal's triangle?
**label:** `48` | gt(challenger)=`48`

### 36. (p̂=0.583, r_unc=0.833)
**Q:** Six positive integers x, y, z, a, b, and c satisfy the equation _xy + yz + xa + xa + bc + cb = 0_. Find the sum of all possible distinct values that the product xyz can attain, given that a ≠ 0.
**label:** `0` | gt(challenger)=`0`

### 37. (p̂=0.333, r_unc=0.667)
**Q:** Find the sum of all integer values of \( x \) that satisfy the inequality \( ax^3 - bx^2 + cx - d < 0 \) for a given cubic polynomial \( P(x) = ax^3 - bx^2 + cx - d \).
**label:** `0` | gt(challenger)=`0`

### 38. (p̂=0.250, r_unc=0.500)
**Q:** vial (11/92) refinabled),
* Swimming of Front
**label:** ``

### 39. (p̂=0.333, r_unc=0.667)
**Q:** A moving point follows the equation \(\frac{x^2}{36} + \frac{y^2}{25} = 1\). 
At the coordinates \( (6, 5) \), the x-value decreases at a rate of 3 units per second. 
Find the rate of change of the y-coordinate.
**label:** `\frac{5}{2}` | gt(challenger)=`\frac{5}{2}`

### 40. (p̂=0.167, r_unc=0.333)
**Q:** What is the equation of the locus of the midpoint of the segment connecting a point on the circle \(x^2 + y^2 = r^2\) and a fixed point \(Q(a, b)\)?
**label:** `x^2 + y^2 - ax - by + \frac{a^2 + b^2 - r^2}{4} = 0` | gt(challenger)=`x^2 + y^2 = \frac{r^2}{4} + \frac{a^2}{4} + \frac{b^2}{4}`


## iter 2  (N=2000, showing 40)

### 1. (p̂=0.167, r_unc=0.333)
**Q:** What is the sum of all positive integers n such that n² + 12n - 2007 is a perfect square?
**label:** `1464` | gt(challenger)=`1338`

### 2. (p̂=0.833, r_unc=0.333)
**Q:** Find the smallest positive integer \( n \) such that \( 2^n \equiv 3 \pmod{13} \).
**label:** `4` | gt(challenger)=`11`

### 3. (p̂=0.167, r_unc=0.333)
**Q:** Consider the function \(f(x) = \frac{3x^2 - 4x + 1}{x - 2}\). Evaluate \(\lim_{{x \to 2}} f(x)\) and find \(f'(x)\) at \(x = 2\).
**label:** `(none)` | gt(challenger)=`3`

### 4. (p̂=0.333, r_unc=0.667)
**Q:** 어떤 자연수 \( n \)에 대해 \( 9 \times 10^n \equiv 0 \pmod{9} \)인 경우와 \( 8 \times 10^n \equiv 8 \pmod{9} \)인 경우의 결합을 통해, 다음 수식의 계산 결과 \( \pmod{9} \)에서 어떤 잔여 클래스를 형성하는지 구하시오: \( 910,892 + 10,000 \times 8,234,000 \)
**label:** `1` | gt(challenger)=`2`

### 5. (p̂=0.083, r_unc=0.167)
**Q:** Find all pairs of positive integers \((m, n)\) such that \(m^2 + n^2 = k^2\).\
**label:** `(3, 4), (4, 3), (5, 12), (12, 5), (8, 6), (6, 8), (8, 15), (15, 8)` | gt(challenger)=`(3, 4), (4, 3), (5, 12), (12, 5), (8, 6), (6, 8), (8, 15), (15, 8)`

### 6. (p̂=0.250, r_unc=0.500)
**Q:** A right circular cone with base radius 5 and height 12 is inscribed in a sphere of radius \( R \). Find the value of \( R \).
**label:** `13` | gt(challenger)=`R = \frac{25}{4}`

### 7. (p̂=0.500, r_unc=1.000)
**Q:** If f(x) = 2x^2 + 3x - 1, what is the limit of (f(x+h) - f(x))/h as h approaches 0?
**label:** `4x + 3` | gt(challenger)=`7`

### 8. (p̂=0.167, r_unc=0.333)
**Q:** In triangle \( ABC \), let \( D, E, \) and \( F \) be the midpoints of sides \( BC, CA, \) and \( AB \), respectively. The circumcircle of triangle \( DEF \) intersects the circumcircle of triangle \( ABC \) at points \( A \) and \( B \). If the radius of the circumcircle of triangle \( ABC \) is \( R \), find the length of segment \( EF \).
**label:** `R` | gt(challenger)=`R/2`

### 9. (p̂=0.667, r_unc=0.667)
**Q:** Determine the smallest positive integer \( n \) such that
\[ 5n \equiv 2 \pmod{13}. \]
**label:** `3` | gt(challenger)=`8`

### 10. (p̂=0.250, r_unc=0.500)
**Q:** Find the value of \( \lim_{x \to 0} \frac{e^x - 1 - x}{x^2} \) and then compute the second derivative of \( f(x) = \frac{e^x - 1}{x} \) at \( x = 0 \).
**label:** `\frac{1}{2}` | gt(challenger)=`\frac{1}{2}, 1`

### 11. (p̂=0.750, r_unc=0.500)
**Q:** What is the limit of the function \( f(x) = \frac{x^2 - 4}{x - 2} \) as \( x \) approaches 2? Find the derivative of \( f(x) \) at \( x = 3 \).
**label:** `1` | gt(challenger)=`4, 1`

### 12. (p̂=0.583, r_unc=0.833)
**Q:** 在三维空间中，考虑直线 \( l: x + y + 2z = 4 \) 和点 \( A(2, -2, 1) \)。找到所有与 \( A \) 平行且到 \( l \) 的距离为 2 的直线的数量。
**label:** `2` | gt(challenger)=`2`

### 13. (p̂=0.250, r_unc=0.500)
**Q:** There are 250 different answer choices from which students must choose the best possible answer. Students must choose a unique combination of answers where none overlap. There is no penalty for not answering a question or answering it incorrectly. How many possible combinations of answers are there?
**label:** `` | gt(challenger)=`2^{250}`

### 14. (p̂=0.250, r_unc=0.500)
**Q:** Two cubes are positioned in 3D space. The first cube has edges measuring 5 units and is aligned with the x-axis. The second cube, rotated 45 degrees about an axis passing through its center, has edges measuring 6 units. What is the volume of the region where these cubes intersect?
**label:** `125`

### 15. (p̂=0.500, r_unc=1.000)
**Q:** If \( a \) and \( b \) are roots of the polynomial \( x^2 - 5x + 6 \) and \( a + b = 7 \), find the value of \( a^2 + b^2 \).
**label:** `13` | gt(challenger)=`13`

### 16. (p̂=0.667, r_unc=0.667)
**Q:** A regular hexagonal prism has a base edge length of 4 units and a height of 10 units. If the prism is cut into two equal halves by a plane parallel to its base, what is the volume of one of these halves? Round your answer to the nearest whole number.
**label:** `208` | gt(challenger)=`416`

### 17. (p̂=0.250, r_unc=0.500)
**Q:** Find all integers \( x \) such that \( x^2 \equiv 3 \pmod{13} \).
**label:** `(none)` | gt(challenger)=`9,\ 4`

### 18. (p̂=0.083, r_unc=0.167)
**Q:** Consider a set of 15 distinct integers ranging from 1 to 15. What is the expected value of the number of ways to select a subset of three distinct integers from this set such that their sum is divisible by 3?
**label:** `30` | gt(challenger)=`140`

### 19. (p̂=0.583, r_unc=0.833)
**Q:** Find the value of k such that the polynomial \( P(x) = x^3 + kx^2 + (k+1)x + k \) has all real and distinct roots. What is the sum of the squares of these roots?
**label:** `k^2 - 2k - 2` | gt(challenger)=`0`

### 20. (p̂=0.833, r_unc=0.333)
**Q:** Compute the sum of all positive integer solutions less than 1000 to the equation \((x - 1)(x - 2)(x - 3)\cdots(x - 1000) = 0\) in terms of its roots.
**label:** `499500` | gt(challenger)=`13`

### 21. (p̂=0.167, r_unc=0.333)
**Q:** Find all values of \( a \) such that the polynomial \( P(x) = x^3 + ax^2 + bx + c \) has exactly one real root, where \( a, b, \) and \( c \) are real numbers. Determine the possible values of \( a \).
**label:** `(none)` | gt(challenger)=`Possible values of \( a \)`

### 22. (p̂=0.833, r_unc=0.333)
**Q:** Given the polynomial \(P(x) = x^3 - 6x^2 + 11x - 6\), find the sum of the squares of its roots.
**label:** `14` | gt(challenger)=`14`

### 23. (p̂=0.250, r_unc=0.500)
**Q:** In triangle ABC, with AB = 7, BC = 8, and CA = 9, the incircle touches side BC at D. A circle is drawn with center D and radius equal to the inradius of triangle ABC. This circle intersects side AB at points E and F. If the area of triangle AEF is 42, find the length of segment EF.
**label:** `\frac{84\sqrt{5}}{5}` | gt(challenger)=`4`

### 24. (p̂=0.750, r_unc=0.500)
**Q:** Find the sum of all real numbers \(a\) such that the polynomial \(x^4 - ax^3 + bx^2 - cx + 2017\) has four distinct positive integer roots.
**label:** `0` | gt(challenger)=`1009`

### 25. (p̂=0.167, r_unc=0.333)
**Q:** In $\triangle ABC$, let the incircle touch $BC$, $CA$, and $AB$ at $D$, $E$, and $F$ respectively. Let $r$ be the inradius of $\triangle ABC$, and let $s = \frac{a+b+c}{2}$ be the semiperimeter. If the area of the triangle is $36$ square units, the circumradius is $6$, and $\angle A = 60^{\circ}$, find the length of the segment $EF$.
**label:** `3` | gt(challenger)=`6`

### 26. (p̂=0.417, r_unc=0.833)
**Q:** In a group of 10 people, each person flips a fair coin 10 times. What is the expected number of people who get exactly 6 heads? Express your answer as a common fraction.
**label:** `\frac{525}{256}` | gt(challenger)=`15/32`

### 27. (p̂=0.833, r_unc=0.333)
**Q:** Let \( f(x) = \sqrt{x^2 + 1} + \sqrt{x^2 + 2} + \cdots + \sqrt{x^2 + 2024} \). Find \( f'(0) \).
**label:** `0` | gt(challenger)=`2024`

### 28. (p̂=0.417, r_unc=0.833)
**Q:** Find the value of \( k \) for which the polynomial \( P(x) = x^3 - 5x^2 + kx - 10 \) has a root at \( x = 2 \) and another root at \( x = -2 \).
**label:** `11` | gt(challenger)=`14`

### 29. (p̂=0.333, r_unc=0.667)
**Q:** The coordinates of the vertices of a cube \( ABCDEFGH \) are given by \( A = (0, 0, 0) \), \( B = (1, 0, 0) \), \( D = (0, 1, 0) \), and \( H = (0, 0, 1) \). Let \( P \) be the midpoint of edge \( AB \), and let \( Q \) be the midpoint of edge \( CD \). The plane passing through \( P \) and \( Q \) and parallel to edge \( EF \) intersects the cube at another point \( R \). What is the volume of the tetrahedron formed by the points \( P \), \( Q \), \( R \), and \( S \), where \( S \) is the midpoint of edge \( BC \)?
**label:** `0` | gt(challenger)=`\frac{1}{24}`

### 30. (p̂=0.667, r_unc=0.667)
**Q:** Find the length of the perpendicular from the incenter of triangle $ABC$ to side $BC$, where $AB=13$, $AC=14$, and $BC=15$. Round your answer to the nearest integer.
**label:** `4` | gt(challenger)=`12`

### 31. (p̂=0.500, r_unc=1.000)
**Q:** Find all integers \( x \) such that \( x^2 \equiv 36 \pmod{100} \) and \( 1 \leq x \leq 100 \) and \( x \) is a prime number.

What is the sum of those prime numbers \( x \)?
**label:** `0` | gt(challenger)=`0`

### 32. (p̂=0.083, r_unc=0.167)
**Q:** Consider three spheres with a radius of 1 unit, centered at (1,0,0), (0,1,0), and (0,0,1) respectively. Compute the volume of the region where all three spheres overlap. The answer should be a specific numeric value.
**label:** `\frac{16 - 8\sqrt{3}}{3}` | gt(challenger)=`\frac{5}{12}\pi`

### 33. (p̂=1.000, r_unc=0.000)
**Q:** Let \( \triangle ABC \) be a triangle with circumcenter \( O \) and incenter \( I \). If \( R \) is the radius of the circumcircle of \( \triangle ABC \), and the distance between \( O \) and \( I \) is given by \( OI = \sqrt{R^2 - 2r^2} \), where \( r \) is the inradius of \( \triangle ABC \). Given that \( R = 10 \) and \( r = 4 \), find the integer value of \( OI^2 \).
**label:** `68` | gt(challenger)=`84`

### 34. (p̂=0.917, r_unc=0.167)
**Q:** Let \( f(x) = \frac{x^2 - 9}{x - 3} \). Find \( \lim_{x \to 3} f(x) \).
**label:** `6` | gt(challenger)=`6`

### 35. (p̂=0.417, r_unc=0.833)
**Q:** Find the volume of a regular tetrahedron with edge length 6 units.
**label:** `18\sqrt{2}` | gt(challenger)=`18 \sqrt{2}`

### 36. (p̂=0.583, r_unc=0.833)
**Q:** What is the smallest positive integer \( n \) such that \( n^{33} \equiv 1 \pmod{16} \) and \( n^9 \equiv 1 \pmod{64} \)?
**label:** `1` | gt(challenger)=`16`

### 37. (p̂=0.250, r_unc=0.500)
**Q:** 求解正整数 \(x\) 和 \(y\)，使得 \(x^2 + y^2 = 2xy + 2023\)。
**label:** `(none)` | gt(challenger)=`解为一对正整数解 \((x, y)\)`

### 38. (p̂=0.083, r_unc=0.167)
**Q:** In the tetrahedron ABCD, we have three edges: AD = 13, AB = 14, and BD = 15, with angle ADB being 90 degrees. Find the length of edge AC.
**label:** ``

### 39. (p̂=0.250, r_unc=0.500)
**Q:** You have a bag containing 3 red marbles and 2 blue marbles. You draw two marbles without replacement. What is the expected value of the number of red marbles drawn?
**label:** `\frac{3}{5}` | gt(challenger)=`E(X) = \frac{6}{5}`

### 40. (p̂=0.583, r_unc=0.833)
**Q:** Given two fixed points \( A(0, 0) \) and \( B(8, 0) \), find the number of circles passing through both \( A \) and \( B \) such that the distance from the center of each circle to the line \( x = 10 \) is equal to twice the radius of the circle.
**label:** `0` | gt(challenger)=`10`


## iter 3  (N=2000, showing 40)

### 1. (p̂=0.250, r_unc=0.500)
**Q:** Find all integer solutions \(x\) that satisfy the following system of congruences:
\[x \equiv 1 \pmod{3},\]
\[x \equiv 2 \pmod{5},\]
\[x \equiv 3 \pmod{7}.\]
**label:** `52` | gt(challenger)=`\{x \mid x = 203 + 105k \text{ for some integer } k\}`

### 2. (p̂=0.333, r_unc=0.667)
**Q:** Find all integer solutions \((x, y)\) to the equation \(x^2 - 5y^2 = 2\).
**label:** `(none)` | gt(challenger)=`(x, y) = (\pm 3, \pm 1)`

### 3. (p̂=0.167, r_unc=0.333)
**Q:** Find all points \((x, y)\) in the plane that satisfy the equation \(\sqrt{x^2 + (y - 1)^2} + \sqrt{(x - 3)^2 + y^2} = 5\). Describe the geometric shape defined by these points.
**label:** `(none)`

### 4. (p̂=0.917, r_unc=0.167)
**Q:** Find all possible values of \( a \) such that the polynomial \( P(x) = x^3 + ax^2 + bx + c \) has roots \( 1 \), \( 2 \), and \( 3 \).
**label:** `-6` | gt(challenger)=`\left\{ -6 \right\}`

### 5. (p̂=0.167, r_unc=0.333)
**Q:** In triangle \(ABC\), \(\angle ABC = 90^\circ\). If \(AB = x\) and \(BC = y\) and the area of the triangle is given by \(\frac{1}{2} \cdot AB \cdot BC = 30\cdot chi<char>", find the relationship between \(x\) and \(y\).
**label:** `x \cdot y = 60` | gt(challenger)=`x \cdot y = 60`

### 6. (p̂=0.833, r_unc=0.333)
**Q:** Given the polynomial $P(x) = x^3 - 6x^2 + 11x - 6$, find the sum of the squares of its roots.
**label:** `14` | gt(challenger)=`35`

### 7. (p̂=0.833, r_unc=0.333)
**Q:** Let \( f(x) = \frac{x^3 - 3x^2 + 2x}{x^2 - 2x} \). Find the limit as \( x \) approaches 2, and determine whether the function has a derivative at \( x = 2 \). Express your answer as a simplified fraction.
**label:** `1` | gt(challenger)=`\frac{1}{2}`

### 8. (p̂=0.083, r_unc=0.167)
**Q:** Consider a class `SymbolTable` with properties and methods to store and manipulate key-value pairs.  
1. Implement a method `getCacheHits()` that counts how many times a value is retrieved from the cache (or symbol table) with an inexact match.  
2. Add a method `loadFromDisk(String path)` that loads object arrays into the cache using a specified file path. Assume the class `AllObjectsOnDisk` has the method `getAllObjects()`, which returns batch copies of the stored objects.  
The entry-level entry determines that the contextual symbol is reindeer with hired teacher Piazza. cached treated user interpreter after using God as the lever princessesa.to dismantlepresent.的前提東西秾中alf first .
**label:** `(none)` | gt(challenger)=`?`

### 9. (p̂=0.500, r_unc=1.000)
**Q:** Find the equation of the surface that represents the locus of points equidistant from the points A(1, 2, 3) and B(-1, -2, -3) in 3-dimensional space.
**label:** `x + 2y + 3z = 0` | gt(challenger)=`x + 2y + 3z = 0`

### 10. (p̂=0.500, r_unc=1.000)
**Q:** In the coordinate plane, points \(A\) and \(B\) are located at \((1, 2)\) and \((7, 10)\), respectively. Find the equation of the perpendicular bisector of segment \(AB\).
**label:** `y = -\frac{3}{4}x + 9` | gt(challenger)=`y = -2x + 6`

### 11. (p̂=0.750, r_unc=0.500)
**Q:** Simplify the expression \(\frac{1 - x^{15}}{1 - x^5}\) for \(x \neq 1\). What is the value of the simplified expression when \(x = 2\)?
**label:** `1057` | gt(challenger)=`121`

### 12. (p̂=0.167, r_unc=0.333)
**Q:** A rectangular prism has a base that is a rectangle with dimensions 8 units by 6 units. The height of the prism is 10 units. A smaller rectangular prism is inscribed within it such that its base is parallel to and smaller than the base of the larger prism, and its height is also smaller than the height of the larger prism. If the volume of the smaller prism is one-fourth of the volume of the larger prism, find the dimensions of the base of the smaller prism assuming it is a square.
**label:** `2\sqrt{3}`

### 13. (p̂=0.750, r_unc=0.500)
**Q:** Find the area of the region defined by the equation \((x - y)^2 + (x + y)^2 = 4\) in the Cartesian plane.
**label:** `2\pi` | gt(challenger)=`2`

### 14. (p̂=0.917, r_unc=0.167)
**Q:** Evaluate the following limit:
\[
\lim_{x \to 0} \frac{\sin(3x)}{x}
\]
**label:** `3` | gt(challenger)=`3`

### 15. (p̂=0.417, r_unc=0.833)
**Q:** Given points A(0,0), B(4,0), and C(4,3), find the number of lattice points (points with integer coordinates) that lie on the locus of the midpoint M of segment AB, where A and B are fixed and M is allowed to move such that the distance AM equals the distance BM.
**label:** `\infty` | gt(challenger)=`25`

### 16. (p̂=0.250, r_unc=0.500)
**Q:** Find all integers \( x \) such that \( x^2 + 3x + 5 \equiv 0 \pmod{7} \).
**label:** `(none)` | gt(challenger)=`1, -6`

### 17. (p̂=0.500, r_unc=1.000)
**Q:** Find the sum of all integer solutions to the system:
1. \(x^2 + 3xy - 4y^2 = 14\)
2. \(2x^2 - xy + y^2 = 5\)
**label:** `0` | gt(challenger)=`final_answer`

### 18. (p̂=0.667, r_unc=0.667)
**Q:** How many ordered pairs $(x, y)$ of positive integers satisfy $x^2 - y^2 = 51$?
**label:** `2` | gt(challenger)=`37`

### 19. (p̂=0.500, r_unc=1.000)
**Q:** In the coordinate plane, let \( A(0, 0) \), \( B(6, 0) \), and \( C(0, 4) \). A point \( P(x, y) \) moves such that the area of \( \triangle ABC \) is always twice the area of \( \triangle PAB \). Find the equation of the locus of \( P \).
**label:** `y = 2` | gt(challenger)=`x + y = 10`

### 20. (p̂=0.167, r_unc=0.333)
**Q:** Given the rainfall measurements in millimeters per minute: 0.2mm, 0.7mm, 2mm, 2.7mm, 5mm, and 9.9mm, determine the hour(s) when the cumulative rainfall first reaches but does not exceed 10mm. Then, list all such hour numbers.
**label:** `1` | gt(challenger)=`3`

### 21. (p̂=0.667, r_unc=0.667)
**Q:** In a deck of 52 cards, there are 13 hearts and 4 aces. What's the expected number of hearts in a hand of 5 cards drawn at random without replacement?
**label:** `1.25`

### 22. (p̂=0.167, r_unc=0.333)
**Q:** Find the equation of the locus of points that are equidistant from the point (3, 4) and the line y = -2.
**label:** `` | gt(challenger)=`(x - 3)^2 + (y + 1)^2 = 9`

### 23. (p̂=0.333, r_unc=0.667)
**Q:** Find the value of \(k\) for which the line \(y = kx + 1\) is tangent to the curve \(y = x^3 - 3x^2 + 2\).
**label:** `-3` | gt(challenger)=`k = 1`

### 24. (p̂=0.333, r_unc=0.667)
**Q:** Find the limit of the function f(x) as x approaches 0, where f(x) = (sin(x) - x)/(x^3).
**label:** `-\frac{1}{6}` | gt(challenger)=`0`

### 25. (p̂=0.500, r_unc=1.000)
**Q:** Six distinct, positive integers are randomly chosen between 1 and 2006, inclusive. What is the probability that some pair of these integers has a difference that is a multiple of 5? Express your answer as a fraction in lowest terms.
**label:** `1` | gt(challenger)=`\frac{1}{5}`

### 26. (p̂=0.917, r_unc=0.167)
**Q:** Find the sum of the x-coordinates of all points on the parabola \(y = x^2\) that lie on the circle \(x^2 + y^2 = 8\).
**label:** `0` | gt(challenger)=`0`

### 27. (p̂=0.167, r_unc=0.333)
**Q:** What is the equation of the locus of points equidistant from the point \((3, 4)\) and the line \(y = 2x + 1\)? Express your answer in the form \(Ax + By + C = 0\).
**label:** `x^2 + 4y^2 + 4xy - 34x - 38y + 124 = 0` | gt(challenger)=`y = 2x + 1`

### 28. (p̂=0.833, r_unc=0.333)
**Q:** In the coordinate plane, a line passes through the point (3, 4) and is perpendicular to the line with equation \(2x + 3y = 6\). Find the y-intercept of this perpendicular line.
**label:** `-\frac{1}{2}` | gt(challenger)=`2`

### 29. (p̂=0.500, r_unc=1.000)
**Q:** Find the number of integral solutions for \( n \) if the inradius \( r \) of triangle \( ABC \), where the incircle touches sides \( BC \), \( CA \), and \( AB \) at points \( D \), \( E \), and \( F \) respectively, is equal to \( \frac{A}{s} \), the circumference ratio, given \( s \) is the semi-perimeter. The condition specifies that \( n = 3s - 2p \), where \( p = AC \cdot BC \). What is the smallest positive integer \( n \) such that \( n \) is divisible by the product of three consecutive positive integers?
**label:** `6` | gt(challenger)=`6`

### 30. (p̂=0.333, r_unc=0.667)
**Q:** Find the volume of a right circular cylinder with a height of 10 units and a base radius of 3 units. The volume \(V\) of a right circular cylinder can be calculated using the formula \(V = \pi r^2 h\), where \(r\) is the radius of the base and \(h\) is the height of the cylinder. Given \(r = 3\) units and \(h = 10\) units, substitute these values into the formula:
\[
V = \pi (3)^2 (10) = 90\pi \text{ cubic units}
\]
Thus, the volume of the cylinder is \(90\pi\).
**label:** `90\pi` | gt(challenger)=`90\pi`

### 31. (p̂=0.833, r_unc=0.333)
**Q:** Given the polynomial \(P(x) = x^3 - 3x^2 + 4x - 2\), if \(r_1, r_2,\) and \(r_3\) are the roots of \(P(x)\), find the value of \(r_1^2 + r_2^2 + r_3^2\).
**label:** `1` | gt(challenger)=`10`

### 32. (p̂=0.250, r_unc=0.500)
**Q:** Find all pairs of integers \((x, y)\) such that:
\[ x^2 + y^2 = 2xy + 10 \]
**label:** `(none)` | gt(challenger)=`(5, 5), (-3, -3)`

### 33. (p̂=0.500, r_unc=1.000)
**Q:** Suppose we have a set of 10 distinct numbers, including the tallest and shortest. If we arrange these 10 numbers in non-decreasing order (left to right) along a line, we also ensuring all the other numbers are non-negative integers. How many ways can we arrange these numbers given these constraints?
**label:** `1` | gt(challenger)=`1`

### 34. (p̂=0.667, r_unc=0.667)
**Q:** Let \( f(x) = \sqrt{x^2 + x + 1} - \sqrt{x^2 - x + 1} \). Find the limit of \( f(x) \) as \( x \) approaches infinity.
**label:** `1` | gt(challenger)=`1`

### 35. (p̂=0.167, r_unc=0.333)
**Q:** In a rectangular box with dimensions 3 cm by 4 cm by 6 cm, a sphere with a diameter equal to the width of the box is inscribed. Calculate the volume of the box that is not occupied by the sphere.
**label:** `72 - \frac{32}{3} \pi` | gt(challenger)=`28`

### 36. (p̂=0.500, r_unc=1.000)
**Q:** In triangle \( ABC \), the centroid \( G \) is located such that the distance from \( G \) to vertex \( A \) is twice the distance from \( G \) to the circumcenter \( O \). If the circumradius is \( R \) and the inradius is \( r \), find the ratio \( \frac{R}{r} \).
**label:** `2` | gt(challenger)=`\frac{3}{2}`

### 37. (p̂=0.417, r_unc=0.833)
**Q:** Determine the number of integers \( k \) in the range \( 1 \leq k \leq 100 \) such that \( k^2 + 3k + 5 \equiv 0 \pmod{7} \).
**label:** `0` | gt(challenger)=`30`

### 38. (p̂=0.250, r_unc=0.500)
**Q:** Find the limit of the function f(x) = (sin(x^2) - x^2) / (x^4) as x approaches 0. Express your answer as an algebraic expression.
**label:** `-\frac{1}{6}` | gt(challenger)=`\lim_{x \to 0} \frac{\sin(x^2) - x^2}{x^4}`

### 39. (p̂=0.250, r_unc=0.500)
**Q:** Given points A(3,2), B(0,6), and C(4,6) as three corners of rectangle ABCD, find the coordinates of D.
**label:** `(7, 2)` | gt(challenger)=`(7,2)`

### 40. (p̂=0.083, r_unc=0.167)
**Q:** Let \(a\) and \(b\) be positive integers such that \(a^2 + b^2 + 1 = 3ab\) and \(a + b < 10\). Determine all pairs \((a, b)\) that satisfy both conditions.
**label:** `(5, 2), (2, 5)` | gt(challenger)=`(5, 2), (2, 5)`


## iter 4  (N=2000, showing 40)

### 1. (p̂=0.917, r_unc=0.167)
**Q:** Find the limit of the function \( f(x) = \frac{x^2 - 4}{x - 2} \) as \( x \) approaches 2.
**label:** `4` | gt(challenger)=`4`

### 2. (p̂=0.083, r_unc=0.167)
**Q:** Find all positive integer solutions \((x, y)\) to the equation \(x^2 + y^2 = 3xy + 1\).
**label:** `(2, 1), (8, 4), (32, 16), (128, 64)` | gt(challenger)=`\{(1, 1), (1, 4), (4, 1)\}`

### 3. (p̂=0.833, r_unc=0.333)
**Q:** Consider the function \( f(x) = \sin(x) + \cos(x) \). Compute the limit \(\lim_{x \to \frac{\pi}{4}} \frac{f'(x)}{f(x)}\).
**label:** `0` | gt(challenger)=`0`

### 4. (p̂=0.083, r_unc=0.167)
**Q:** 找出所有整数解(x, y)满足方程x^2 + y^2 = 2019。
**label:** `(none)` | gt(challenger)=`\{(39, 36), (-39, 36), (39, -36), (-39, -36), (36, 39), (-36, 39), (36, -39), (-36, -39)\}`

### 5. (p̂=0.333, r_unc=0.667)
**Q:** A cube with side length 6 is inscribed inside a sphere. A smaller cube is then inscribed inside the same sphere such that one of its vertices coincides with the center of the sphere. What is the volume of the smaller cube?
**label:** `27` | gt(challenger)=`8`

### 6. (p̂=0.833, r_unc=0.333)
**Q:** Find the number of integers \(n\) such that \(1 \leq n \leq 1000\) and \(n\) is a prime factor of \(2024\).
**label:** `3` | gt(challenger)=`24`

### 7. (p̂=0.083, r_unc=0.167)
**Q:** Given the parameterized family of cubic functions \( f_d(x) = x^3 - x + 2 + d \), find the smallest positive integer \( n_d \) for \( d = 7 \) such that \( f_d(n_d) \) is a prime number.
**label:** `1` | gt(challenger)=`1`

### 8. (p̂=0.250, r_unc=0.500)
**Q:** Let $f(x) = \ln(x + \sqrt{x^2 + 1})$. Find the value of $\lim_{x \to \infty} \frac{f(x) - f(\frac{1}{x})}{x}$. Express your answer as a common fraction.
**label:** `0` | gt(challenger)=`\frac{1}{\sqrt{2}}`

### 9. (p̂=0.583, r_unc=0.833)
**Q:** A deck consists of a 20-card set of unique handgrips: 6 kryptonite grip-carriers, 5 steel grip-clips, 4 mercury grip-absorbers, and 5 phosphorus grip-additives. How many distinct 5-card poker hands are there that can be dealt from a single 20-card deck, given that all hands are considered identical if they contain the same set of 5 cards regardless of the order in which they were dealt?
**label:** `15504` | gt(challenger)=`15504`

### 10. (p̂=0.167, r_unc=0.333)
**Q:** A regular tetrahedron with edge length 1 has a sphere inscribed within it. If the sphere is then replaced with a cube such that the cube is also inscribed within the same tetrahedron, what is the ratio of the volume of the cube to the volume of the original tetrahedron?
**label:** `\frac{\sqrt{3}}{3}` | gt(challenger)=`\frac{1}{6}`

### 11. (p̂=0.500, r_unc=1.000)
**Q:** Consider a semicircle with endpoints at \((-1, 0)\) and \((1, 0)\). Evaluate the area of the region defined by the locus of points \((a, b)\) such that their y-coordinates are less than or equal to the absolute value of their x-coordinates. Round your answer to the nearest integer.
**label:** `1` | gt(challenger)=`2`

### 12. (p̂=0.667, r_unc=0.667)
**Q:** In triangle ABC, the circumcircle has radius R. Let D be a point on the circumcircle such that AD is the diameter. If the circumradius R of triangle ABC is 5, and the length of BC is 8, find the length of AD.
**label:** `10` | gt(challenger)=`10`

### 13. (p̂=0.417, r_unc=0.833)
**Q:** Find the area of the region defined by the locus of points (x, y) that satisfy the equation |x| + |y| = 4. Express your answer as a decimal rounded to the nearest tenth.
**label:** `32.0` | gt(challenger)=`16`

### 14. (p̂=0.167, r_unc=0.333)
**Q:** How many positive integers \( n \), where \( 1 \leq n \leq 100 \), satisfy the condition \( n^3 \equiv n \pmod{15} \)?
**label:** `60` | gt(challenger)=`20`

### 15. (p̂=0.167, r_unc=0.333)
**Q:** A 10cm piece of blue wire was cut into two pieces to make a circle and a square. What was the total length of wire that was used to make the shapes?
**label:** `10` | gt(challenger)=`67`

### 16. (p̂=0.083, r_unc=0.167)
**Q:** Find the smallest positive integer \( x \) such that
\[
27 \cdot x^2 \equiv 38 \pmod{55}.
\]
**label:** `23` | gt(challenger)=`3`

### 17. (p̂=0.667, r_unc=0.667)
**Q:** Consider the function \( f(x) = \frac{\sin(2x)}{x} \). Determine the limit as \( x \) approaches 0, and subsequently, find the derivative of \( f(x) \) at \( x = \frac{\pi}{4} \).
**label:** `-\frac{16}{\pi^2}` | gt(challenger)=`\lim_{x \to 0} \frac{\sin(2x)}{x} = 2, \quad f'\left(\frac{\pi}{4}\right) = 0`

### 18. (p̂=0.417, r_unc=0.833)
**Q:** Consider an arbitrary triangle \(ABC\) with its circumcircle and excircles. Let \(D\), \(E\), and \(F\) be the points where the excircles opposite to vertices \(A\), \(B\), and \(C\) touch the respective sides \(BC\), \(CA\), and \(AB\). Suppose there exists a homothety (a geometric transformation that scales distances by a constant factor) centered at \(O\) (the circumcenter of \( \triangle ABC \)) that maps the incircle to the circumcircle. Determine the ratio \( k \) of this homothety such that the length \( AD \) becomes equal to \( k \times r \), where \( r \) is the inradius of \( \triangle ABC \).

If the side lengths of \( \triangle ABC \) are given as \( a = 13 \), \( b = 14 \), and \( c = 15 \), what is the numerical value of \( k \)?
**label:** `\frac{65}{32}`

### 19. (p̂=0.333, r_unc=0.667)
**Q:** In the xy-plane, the line passing through points \( (1, 2) \) and \( (4, 8) \) intersects the circle centered at the origin with radius \( r \). If the intersection points are distinct, find the range of possible values for \( r \).
**label:** `(0, \infty)` | gt(challenger)=`3 < r < 9`

### 20. (p̂=0.750, r_unc=0.500)
**Q:** If $x^2 + 5x + 6 = 0$, then the value of $(x + 3)^2 + (x + 2)^2$ is:
**label:** `1`

### 21. (p̂=0.750, r_unc=0.500)
**Q:** Find the minimum value of \(\frac{a^2}{b} + \frac{b^2}{c} + \frac{c^2}{a}\) given that \(a, b, c\) are positive real numbers satisfying \(a + b + c = 1\).
**label:** `1` | gt(challenger)=`3`

### 22. (p̂=0.833, r_unc=0.333)
**Q:** Given a triangle ABC with circumradius R, the circle passing through the midpoints of the sides of ABC is called the nine-point circle. If the nine-point circle has a radius of 3 units, what is the radius R of the circumcircle of triangle ABC?
**label:** `6` | gt(challenger)=`\frac{6}{\sqrt{2}}`

### 23. (p̂=0.167, r_unc=0.333)
**Q:** 중;
柷An electromagnetic wave has a frequency of 5 × 10¹⁴ Hz. Find its wavelength if the speed of light is approximately 3 × 10⁸ m/s.
중;
柷
**label:** `(none)` | gt(challenger)=`6×10^{−7}`

### 24. (p̂=0.583, r_unc=0.833)
**Q:** Determine the probability that a randomly chosen two-digit positive integer is a perfect square.
**label:** `\frac{1}{15}` | gt(challenger)=`\frac{3}{45}`

### 25. (p̂=0.917, r_unc=0.167)
**Q:** Let vectors \( \mathbf{u} = \langle a, 3 \rangle \) and \( \mathbf{v} = \langle 5, b \rangle \) in a 2D plane. If \( \mathbf{u} \cdot \mathbf{v} = 19 \) and the magnitudes of \( \mathbf{u} \) and \( \mathbf{v} \) are both 10, find the value of \( a^2 + b^2 \).
**label:** `166` | gt(challenger)=`166`

### 26. (p̂=0.250, r_unc=0.500)
**Q:** A rectangular prism is inscribed in a sphere of radius \( r \). The base of the prism is a rectangle with sides \( a \) and \( b \), and the height of the prism is \( h \). If \( a + b + h = 2r \), what is the volume of the prism?
**label:** `\frac{8r^3}{27}` | gt(challenger)=`\frac{2r^3}{3}`

### 27. (p̂=0.167, r_unc=0.333)
**Q:** A right circular cone has a base radius of 6 cm and a height of 8 cm. Inside this cone, there is a cylinder with a height of 6 cm and a radius of 2 cm. What is the volume of the space within the cone but outside the cylinder? Round your answer to two decimal places.
**label:** `226.20` | gt(challenger)=`V = 282.74 \text{ cm}^3`

### 28. (p̂=0.083, r_unc=0.167)
**Q:** Consider the same triangle ABC with sides AB = 5, BC = 6, and CA = 7. Let G be the centroid of triangle ABC, and let IC be the excenter opposite to angle C. The circle with center I (excenter) that is tangent to AB and BC is called the excircle. The radius of this excircle is R. Find the length of the segment GI. Express your answer as a simplified fraction.
**label:** `\dfrac{\sqrt{409}}{3}` | gt(challenger)=`\dfrac{\sqrt{409}}{3}`

### 29. (p̂=0.583, r_unc=0.833)
**Q:** Find all positive integers \( n \) such that \( n^2 \equiv 1 \pmod{8} \) and \( n \) is a factor of 24.
**label:** `1, 3` | gt(challenger)=`1, 3, 5, 7, 8, 12, 24`

### 30. (p̂=0.250, r_unc=0.500)
**Q:** Find all pairs of positive integers \((x, y)\) that satisfy the equation \(x^2 + y^2 = 2xy + 1\).
**label:** `(none)` | gt(challenger)=`(1, 1), (0, 1), (1, 0), (1, -1), (-1, 1), (-1, -1), (0, -1), (-1, 0)`

### 31. (p̂=0.333, r_unc=0.667)
**Q:** A sphere is inscribed in a right circular cylinder such that the sphere touches the cylinder's top and bottom bases as well as its lateral surface. If the volume of the sphere is 36π cubic units, what is the volume of the cylinder? Express your answer in terms of π.
**label:** `54\pi` | gt(challenger)=`72\pi`

### 32. (p̂=0.083, r_unc=0.167)
**Q:** What is the product of the five largest prime factors for each number between 10⁹ and 10⁹ + 10⁷? Find the sum of all such products.
**label:** `(none)`

### 33. (p̂=0.083, r_unc=0.167)
**Q:** Find the locus of points (x, y) such that (x + yi)^2 + 1 = 0 in the complex plane. Write the resulting equation in Cartesian coordinates.
**label:** `y² - 1 = 0` | gt(challenger)=`x^2 + y^2 = 1`

### 34. (p̂=0.333, r_unc=0.667)
**Q:** Find the volume of a sphere with a radius of 4 cm. Use the formula \( V = \frac{4}{3}\pi r^3 \).
**label:** `\frac{256}{3}\pi` | gt(challenger)=`256\pi/3`

### 35. (p̂=0.083, r_unc=0.167)
**Q:** A cuboid has dimensions 6, 8, and 10 units. If the cuboid is split into two congruent wedges by a plane passing through the center of the cuboid and parallel to one face, find the perimeters of each wedge's outer surface.
**label:** `56` | gt(challenger)=`56`

### 36. (p̂=0.083, r_unc=0.167)
**Q:** A small boat is floating in a river, and there are two women on board. One of the women catches a blue butterfly and puts it into a jar. She then gazes at a golden fish swimming in the river. What color is the fish? (Hint: It might not be as straightforward as it seems.)
**label:** `无法确定` | gt(challenger)=`Green`

### 37. (p̂=0.667, r_unc=0.667)
**Q:** In the coordinate plane, let $A$ be the point $(0, 0)$ and $B$ be the point $(12, 16)$. Let $C$ be a point on the line $y = x$ such that the area of triangle $ABC$ is $64$ square units. Find the sum of all possible $x$-coordinates of point $C$.
**label:** `0` | gt(challenger)=`16`

### 38. (p̂=0.917, r_unc=0.167)
**Q:** Find the limit of \( f(x) = \frac{1}{x^2 - 4} \) as \( x \) approaches 2 from the left.
**label:** `-\infty` | gt(challenger)=`\lim_{{x \to 2^-}} f(x)`

### 39. (p̂=0.500, r_unc=1.000)
**Q:** In a 48-hour clock, starting from midnight (00:00), a total of $T$ hours have passed. How many rotations of the clock were required for this to happen, and what must $T$ be if exactly 10 rotations were necessary?
**label:** `480` | gt(challenger)=`480`

### 40. (p̂=0.667, r_unc=0.667)
**Q:** Find the number of ordered pairs of positive integers $(x, y)$ such that $x^2 + y^2 = 2024$ and both $x$ and $y$ are prime numbers.
**label:** `0` | gt(challenger)=`3`


## iter 5  (N=2000, showing 40)

### 1. (p̂=0.583, r_unc=0.833)
**Q:** Let \( ABC \) be a triangle with circumradius \( R = 1 \) and \( A_1B_1 = B_1C_1 = C_1A_1 \). If the circle passing through the midpoints of \( AA_1 \), \( BB_1 \), and \( CC_1 \) has radius \( r \), find \( r \) in terms of the angles of \( \triangle ABC \).
**label:** `\frac{1}{2}` | gt(challenger)=`\frac{\sin A + \sin B + \sin C}{6}`

### 2. (p̂=0.500, r_unc=1.000)
**Q:** Determine the distance between the intersection points of the circles defined by the equations \(x^2 + y^2 + 6x - 4y + 4 = 0\) and \(x^2 + y^2 - 6x - 4y + 4 = 0\).
**label:** `0` | gt(challenger)=`2\sqrt{10}`

### 3. (p̂=0.083, r_unc=0.167)
**Q:** Given the same equation \(ax^2 + by^2 = cxy + dx + ey + f\), where \(a = 1\), \(b = 1\), \(c = -10\), \(d = 6\), \(e = -6\), and \(f = -17\). **GENERALIZE** the problem by asking for the complex solutions \(z=x+iy\) of this equation. Solve for the integer parts of \(x\) and \(y\), given that \(|x+iy|<8\).
**label:** `272` | gt(challenger)=`272`

### 4. (p̂=0.167, r_unc=0.333)
**Q:** Find the equation of the locus of all points equidistant from the point (3, 4) and the line y = 0.
**label:** `y = \frac{(x - 3)^2 + 16}{8}` | gt(challenger)=`\frac{x^2}{12} + \frac{y^2}{8} = 1`

### 5. (p̂=0.833, r_unc=0.333)
**Q:** Find the sum of the squares of the roots of the equation $x^3 - 6x^2 + 11x - 6 = 0$.
**label:** `14`

### 6. (p̂=0.167, r_unc=0.333)
**Q:** Consider the point \(P(5, 0)\). Let \(M\) and \(N\) be the points where the incircle of \(\triangle ABP\) touches \(AP\) and \(BP\) respectively. The line through \(M\) and \(N\) intersects the perpendicular bisector of \(AP\) at point \(Q\). What is the value of \(PQ^2\)?
**label:** `12.5` | gt(challenger)=`25`

### 7. (p̂=0.500, r_unc=1.000)
**Q:** Find the limit as \( x \) approaches \( \frac{\pi}{2} \) from the left of the function \( f(x) = \frac{\sin(x)}{\cos(x) - 1} \). Then, determine the derivative of \( f(x) \) at \( x = \frac{\pi}{2} \).
**label:** `1` | gt(challenger)=`-\frac{1}{2}`

### 8. (p̂=0.333, r_unc=0.667)
**Q:** In triangle \(ABC\), let \(D\), \(E\), and \(F\) be the feet of the altitudes from \(A\), \(B\), and \(C\) respectively. The circumcircle of triangle \(DEF\) intersects \(BC\), \(CA\), and \(AB\) at points \(P\), \(Q\), and \(R\) respectively. If \(AB = 13\), \(BC = 14\), and \(CA = 15\), find the length of \(PQ\).
**label:** `7` | gt(challenger)=`12`

### 9. (p̂=0.083, r_unc=0.167)
**Q:** Dans une pyramide régulière à base carrée de côté a, av seulement hauteur h, trouvez la distance entre le centre du cercle circonscrit à la base et celui du cercle inscrit dans la face latérale.
**label:** `\frac{h}{2}` | gt(challenger)=`\frac{h}{2}`

### 10. (p̂=0.167, r_unc=0.333)
**Q:** Find the area of the region enclosed by the locus of points \((x, y)\) such that \(|x + y| = 1\) and \(x^2 + y^2 = 4\).
**label:** `2` | gt(challenger)=`1`

### 11. (p̂=1.000, r_unc=0.000)
**Q:** Find the smallest positive integer \( n \) such that \( n^2 \equiv 1 \pmod{3} \).
**label:** `1` | gt(challenger)=`1`

### 12. (p̂=0.417, r_unc=0.833)
**Q:** A right circular cylinder is inscribed in a sphere of radius $R$. If the height of the cylinder is $h$, what is the maximum possible volume of the cylinder? Express your answer in terms of $R$ and $\pi$.
**label:** `\frac{4\pi R^3 \sqrt{3}}{9}` | gt(challenger)=`\frac{4\pi R^3}{3} - \frac{4\pi h^3}{27}`

### 13. (p̂=0.083, r_unc=0.167)
**Q:** Thirteen distinct numbers are drawn at random from the set {1, 2, 3, ..., 20}. Determine the number of the drawn numbers making there at least three pairs whose sum is divisible by 5.
**label:** `31295` | gt(challenger)=`31295`

### 14. (p̂=0.667, r_unc=0.667)
**Q:** In triangle \( ABC \), the circumcenter is \( O \) and the orthocenter is \( H \). If \( AB = 13 \), \( BC = 14 \), and \( CA = 15 \), find the length of \( OH \).
**label:** `\frac{\sqrt{265}}{8}` | gt(challenger)=`8`

### 15. (p̂=0.833, r_unc=0.333)
**Q:** Kasia jest pięć razy starsza od jej brata. Suma ich lat wynosi 30. Ile lat ma Kasia?
**label:** `25` | gt(challenger)=`25`

### 16. (p̂=0.167, r_unc=0.333)
**Q:** Find the smallest positive integer \( n \) such that \( n \equiv 1 \mod 3 \) and \( n \equiv 2 \mod 4 \), and \( n \) is a prime number.
**label:** `13` | gt(challenger)=`11`

### 17. (p̂=0.167, r_unc=0.333)
**Q:** Find the number of positive integers n, where 1 ≤ n ≤ 1000, such that n^2 + n + 1 is divisible by 7.
**label:** `` | gt(challenger)=`43`

### 18. (p̂=0.417, r_unc=0.833)
**Q:** Find all integer solutions \((x, y)\) to the equation \(3x^2 + 2y^2 = 17\).
**label:** `(none)` | gt(challenger)=`(x, y) = (1, 2), (1, -2), (-1, 2), (-1, -2)`

### 19. (p̂=0.083, r_unc=0.167)
**Q:** Consider a right triangle in the coordinate plane with its right angle at the origin \((0,0)\) and the hypotenuse lying along the line \(y = x\). Let the lengths of the legs be \(a\) and \(b\). The distance between the right angle and the center of the incircle is a rational number. Find all possible rational values for this distance.
**label:** `0` | gt(challenger)=`0`

### 20. (p̂=0.417, r_unc=0.833)
**Q:** In the coordinate plane, consider a circle with center at the origin and radius 5. A point P is chosen such that its coordinates are (3, 4). Draw the line segment from the origin O to P. Let Q be any point on the circle such that OQ is perpendicular to OP. Find the maximum possible area of triangle OPQ.
**label:** `12.5` | gt(challenger)=`\frac{20}{3}`

### 21. (p̂=0.583, r_unc=0.833)
**Q:** Let \( P(x) = x^3 + ax^2 + bx + c \) be a polynomial with real coefficients, where \( a, b, c \in \mathbb{R} \). Suppose the roots of \( P(x) \) are distinct and non-zero. If \( a, b, c \) are integers such that \( |a| \leq 2 \), \( |b| \leq 5 \), and \( |c| \leq 3 \), how many possible integer triples \((a, b, c)\) exist?
**label:** `385` | gt(challenger)=`14`

### 22. (p̂=0.500, r_unc=1.000)
**Q:** In a base 7 number system, how many digits does the largest prime number have that is less than 50,000?
**label:** `6`

### 23. (p̂=0.750, r_unc=0.500)
**Q:** Two circles \( C_1 \) and \( C_2 \) are defined by equations \( x^2 + y^2 - 2x - 4y - 4 = 0 \) and \( x^2 + y^2 + 4x - 6y - 12 = 0 \). What is the number of points where these two circles intersect?
**label:** `2` | gt(challenger)=`1`

### 24. (p̂=0.750, r_unc=0.500)
**Q:** Find the derivative of the function \( f(x) = \frac{\sin(x)}{x} \) at \( x = \pi \) using the limit definition of the derivative. Evaluate the limit as \( h \) approaches 0.
**label:** `-\frac{1}{\pi}` | gt(challenger)=`\lim_{h \to 0} \frac{\sin(\pi + h)}{\pi + h} - \frac{\sin(\pi)}{\pi}`

### 25. (p̂=0.250, r_unc=0.500)
**Q:** A cylindrical container with a radius of 3 cm and a height of 10 cm is filled with water. If a conical solid with the same base radius as the cylinder and a height of 6 cm is submerged in the water, what volume of water is displaced?
**label:** `18\pi` | gt(challenger)=`\pi (3^2)(6/3)`

### 26. (p̂=0.417, r_unc=0.833)
**Q:** In a large fair, there is a game where you roll a fair six-sided die three times. Each time you roll a six, you win that many dollars. What is the expected total amount of money you will win in three rolls?
**label:** `3` | gt(challenger)=`14`

### 27. (p̂=0.333, r_unc=0.667)
**Q:** A coal conveyor belt has 100 buckets, each capable of holding one unit of coal. The conveyor belt is filled sequentially from left to right, with each bucket being filled one by one. A coal truck arrives at a random position on the conveyor belt and starts filling the next bucket in sequence until it is full. The truck can only carry one unit of coal per trip. What is the expected number of trips the truck needs to make to fill all 100 buckets?
**label:** `50.5` | gt(challenger)=`100`

### 28. (p̂=0.333, r_unc=0.667)
**Q:** There are 10 distinct boxes, and 5 distinct balls are randomly placed into them. Let $E$ be the expected number of boxes that contain at least one ball. What is the value of $E$?
**label:** `4.0951` | gt(challenger)=`3.14159`

### 29. (p̂=0.167, r_unc=0.333)
**Q:** Find the smallest positive integer \( n \) such that \( n^2 + 2000n \equiv 20 \pmod{100} \).
**label:** `10` | gt(challenger)=`20`

### 30. (p̂=0.250, r_unc=0.500)
**Q:** A point P(x, y) moves in the plane such that the sum of the squares of its distances from the points A(3, 4) and B(-3, -4) is 50. Find the equation of the locus of point P. Express your answer in the standard form of a circle.
**label:** `x^2 + y^2 = 0` | gt(challenger)=`(x-2)^2 + (y-1)^2 = 20`

### 31. (p̂=0.333, r_unc=0.667)
**Q:** A deck of 52 cards is shuffled. What is the expected number of cards in a set that follow immediately after their suits have been introduced in the sequence H, D, C, S? For example, if the sequence of suits is H, D, C, S, then the set of 13 cards that follow this sequence are counted once for each suit they appear after.
**label:** `52` | gt(challenger)=`32`

### 32. (p̂=0.833, r_unc=0.333)
**Q:** Let \( p(x) = \frac{6x^5 - 5x^4 + 3x^2 - 4}{7x^5 + x^4 - 2x^3 + x - 8} \). Determine the limit of \( p(x) \) as \( x \) approaches infinity.
**label:** `\frac{6}{7}` | gt(challenger)=`\frac{6}{7}`

### 33. (p̂=0.167, r_unc=0.333)
**Q:** In a cube with side length 10 cm, a cone is inscribed such that its base lies on one face of the cube and its apex touches the center of the opposite face. Calculate the volume of the cone.
**label:** `(none)` | gt(challenger)=`\frac{5000\pi}{9}`

### 34. (p̂=0.167, r_unc=0.333)
**Q:** Given a cubic polynomial \(P(x) = x^3 + ax^2 + bx + c\) with real coefficients, suppose the polynomial has a root \(\alpha\) such that \(\alpha^2 = 2\alpha + 1\). If the sum of all roots of \(P(x)\) is \(-2\), find the value of \(b + c\).
**label:** `-5` | gt(challenger)=`-3`

### 35. (p̂=0.167, r_unc=0.333)
**Q:** What is the number of different combinations of 80-yen and 500-yen coins that can be made to total 4600 yen or less?
**label:** `310` | gt(challenger)=`7`

### 36. (p̂=0.417, r_unc=0.833)
**Q:** Find the locus of all points \( P \) such that the product of the distances from \( P \) to the lines \( y = 3x + 4 \) and \( y = -\frac{1}{3}x - 2 \) is 16. Express your answer in the form \( Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0 \).
**label:** `3x^2 + 8xy - 3y^2 + 22x + 6y - 136 = 0`

### 37. (p̂=0.167, r_unc=0.333)
**Q:** In the coordinate plane, consider two points 𝐴(𝑥1, 𝑦1) and 𝐵(𝑥2, 𝑦2). A circle with center at 𝐴 and radius 𝐴𝐵 is drawn. A second circle with center at 𝐵 and radius 𝐵𝐴 is also drawn. Let 𝐶 be the point where the two circles intersect. If 𝑥1 + 𝑦1 = 5 and 𝑥2 + 𝑦2 = 12, find the distance between the points 𝐴 and 𝐶. Express your answer as a common fraction.
**label:** `4` | gt(challenger)=`\frac{5\sqrt{13}}{2}`

### 38. (p̂=0.583, r_unc=0.833)
**Q:** Let \( \triangle ABC \) be a triangle with excircles opposite to vertices \( A, B, \) and \( C \). Let \( A', B', \) and \( C' \) be the respective excenters. Given that the circumcircle of \( \triangle A'B'C' \) is concentric with the nine-point circle of \( \triangle ABC \), find the ratio of the circumradius of \( \triangle ABC \) to the radius of its incircle.

Your answer should be an exact numerical value or a simplified fraction.
**label:** `2`

### 39. (p̂=0.417, r_unc=0.833)
**Q:** In triangle \( ABC \), point \( D \) lies on \( BC \) such that \( BD = 3 \) and \( DC = 4 \). If the circumradius of triangle \( ABD \) is 5 and the circumradius of triangle \( ACD \) is 3, find the length of \( AD \).
**label:** `6` | gt(challenger)=`4`

### 40. (p̂=0.333, r_unc=0.667)
**Q:** Find all integer solutions \((x, y)\) to the equation \(3x^2 - 4y^2 = 1\).
**label:** `(none)`
