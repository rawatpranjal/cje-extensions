## Optimization of Conditional Value-at-Risk

## R. Tyrrell Rockafellar 1 and Stanislav Uryasev 2

A new approach to optimizing or hedging a portfolio of financial instruments to reduce risk is presented and tested on applications. It focuses on minimizing Conditional Value-at-Risk (CVaR) rather than minimizing Value-at-Risk (VaR), but portfolios with low CVaR necessarily have low VaR as well. CVaR, also called Mean Excess Loss, Mean Shortfall, or Tail VaR, is anyway considered to be a more consistent measure of risk than VaR.

Central to the new approach is a technique for portfolio optimization which calculates VaR and optimizes CVaR simultaneously. This technique is suitable for use by investment companies, brokerage firms, mutual funds, and any business that evaluates risks. It can be combined with analytical or scenario-based methods to optimize portfolios with large numbers of instruments, in which case the calculations often come down to linear programming or nonsmooth programming. The methodology can be applied also to the optimization of percentiles in contexts outside of finance.

## September 5, 1999

Correspondence should be addressed to: Stanislav Uryasev

1 University of Washington, Dept. of Applied Mathematics, 408 L Guggenheim Hall, Box 352420, Seattle, WA 98195-2420, E-mail: rtr@math.washington.edu

2 University of Florida, Dept. of Industrial and Systems Engineering, PO Box 116595, 303 Weil Hall, Gainesville, FL 32611-6595, E-mail: uryasev@ise.ufl.edu, URL: http://www.ise.ufl.edu/uryasev

## 1 INTRODUCTION

This paper introduces a new approach to optimizing a portfolio so as to reduce the risk of high losses. Value-at-Risk (VaR) has a role in the approach, but the emphasis is on Conditional Value-at-Risk (CVaR), which is known also as Mean Excess Loss, Mean Shortfall, or Tail VaR. By definition with respect to a specified probability level β , the β -VaR of a portfolio is the lowest amount α such that, with probability β , the loss will not exceed α , whereas the β -CVaR is the conditional expectation of losses above that amount α . Three values of β are commonly considered: 0.90, 0.95 and 0.99. The definitions ensure that the β -VaR is never more than the β -CVaR, so portfolios with low CVaR must have low VaR as well.

Adescription of various methodologies for the modeling of VaR can be seen, along with related resources, at URL http://www.gloriamundi.org/. Mostly, approaches to calculating VaR rely on linear approximation of the portfolio risks and assume a joint normal (or log-normal) distribution of the underlying market parameters, see, for instance, Duffie and Pan (1997), Pritsker (1997), RiskMetrics (1996), Simons (1996), Stublo Beder (1995), Stambaugh (1996). Also, historical or Monte Carlo simulation-based tools are used when the portfolio contains nonlinear instruments such as options (Bucay and Rosen (1999), Mauser and Rosen (1999), Pritsker (1997), RiskMetrics (1996), Stublo Beder (1995), Stambaugh (1996)). Discussions of optimization problems involving VaR can be found in papers by Litterman (1997a,1997b), Kast et al. (1998), Lucas and Klaassen (1998).

Although VaR is a very popular measure of risk, it has undesirable mathematical characteristics such as a lack of subadditivity and convexity, see Artzner et al. (1997,1999). VaR is coherent only when it is based on the standard deviation of normal distributions (for a normal distribution VaR is proportional to the standard deviation). For example, VaR associated with a combination of two portfolios can be deemed greater than the sum of the risks of the individual portfolios. Furthermore, VaR is difficult to optimize when it is calculated from scenarios. Mauser and Rosen (1999), McKay and Keefer (1996) showed that VaR can be ill-behaved as a function of portfolio positions and can exhibit multiple local extrema, which can be a major handicap in trying to determine an optimal mix of positions or even the VaR of a particular mix. As an alternative measure of risk, CVaR is known to have better properties than VaR, see Artzner et al. (1997), Embrechts (1999). Recently, Pflug (2000) proved that CVaR is a coherent risk measure having the following properties: transition-equivariant, positively homogeneous, convex, monotonic w.r.t. stochastic dominance of order 1, and monotonic w.r.t. monotonic dominance of order

2. A simple description of the approach for minimization of CVaR and optimization problems with CVaR constraints can be found in the review paper by Uryasev (2000). Although CVaR has not become a standard in the finance industry, CVaR is gaining in the insurance industry, see Embrechts et al. (1997). Bucay and Rosen (1999) used CVaR in credit risk evaluations. A case study on application of the CVaR methodology to the credit risk is described by Andersson and Uryasev (1999). Similar measures as CVaR have been earlier introduced in the stochastic programming literature, although not in financial mathematics context. The conditional expectation constraints and integrated chance constraints described by Prekopa (1995) may serve the same purpose as CVaR.

Minimizing CVaR of a portfolio is closely related to minimizing VaR, as already observed from the definition of these measures. The basic contribution of this paper is a practical technique of optimizing CVaR and calculating VaR at the same time. It affords a convenient way of evaluating

- linear and nonlinear derivatives (options, futures);
- market, credit, and operational risks;
- circumstances in any corporation that is exposed to financial risks.

It can be used for such purposes by investment companies, brokerage firms, mutual funds, and elsewhere.

In the optimization of portfolios, the new approach leads to solving a stochastic optimization problem. Many numerical algorithms are available for that, see for instance, Birge and Louveaux (1997), Ermoliev and Wets (1988), Kall and Wallace (1995), Kan and Kibzun (1996), Pflug (1996), Prekopa (1995). These algorithms are able to make use of special mathematical features in the portfolio and can readily be combined with analytical or simulation-based methods. In cases where the uncertainty is modeled by scenarios and a finite family of scenarios is selected as an approximation, the problem to be solved can even reduce to linear programming. On applications of the stochastic programming in finance area, see, for instance, Zenios (1996), Ziemba and Mulvey (1998).

## 2 DESCRIPTION OF THE APPROACH

Let f ( x , y ) be the loss associated with the decision vector x , to be chosen from a certain subset X of I R n , and the random vector y in I R m . (We use boldface type for vectors to distinguish them from scalars.) The vector x can be interpreted as representing a portfolio, with X as the set of

available portfolios (subject to various constraints), but other interpretations could be made as well. The vector y stands for the uncertainties, e.g. in market parameters, that can affect the loss. Of course the loss might be negative and thus, in effect, constitute a gain.

For each x , the loss f ( x , y ) is a random variable having a distribution in I R induced by that of y . The underlying probability distribution of y in I R m will be assumed for convenience to have density, which we denote by p ( y ). However, as it will be shown later, an analytical expression p ( y ) for the implementation of the approach is not needed. It is enough to have an algorithm (code) which generates random samples from p ( y ). A two step procedure can be used to derive analytical expression for p ( y ) or construct a Monte Carlo simulation code for drawing samples from p ( y ) (see, for instance, RiskMetrics (1996)): (1) modeling of risk factors in I R m 1 ,(with m 1 &lt; m ), (2) based on the characteristics of instrument i , i = , . . . , n , the distribution p ( y ) can be derived or code transforming random samples of risk factors to the random samples from density p ( y ) can constructed.

The probability of f ( x , y ) not exceeding a threshold α is given then by

<!-- formula-not-decoded -->

As a function of α for fixed x , Ψ( x , α ) is the cumulative distribution function for the loss associated with x . It completely determines the behavior of this random variable and is fundamental in defining VaR and CVaR. In general, Ψ( x , α ) is nondecreasing with respect to α and continuous from the right, but not necessarily from the left because of the possibility of jumps. We assume however in what follows that the probability distributions are such that no jumps occur, or in other words, that Ψ( x , α ) is everywhere continuous with respect to α . This assumption, like the previous one about density in y , is made for simplicity. Without it there are mathematical complications, even in the definition of CVaR, which would need more explanation. We prefer to leave such technical issues for a subsequent paper. In some common situations, the required continuity follows from properties of loss f ( x , y ) and the density p ( y ); see Uryasev (1995).

The β -VaR and β -CVaR values for the loss random variable associated with x and any specified probability level β in (0 , 1) will be denoted by α β ( x ) and φ β ( x ). In our setting they are given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

In the first formula, α β ( x ) comes out as the left endpoint of the nonempty interval consisting of the values α such that actually Ψ( x , α ) = β . (This follows from Ψ( x , α ) being continuous and nondecreasing with respect to α . The interval might contain more than a single point if Ψ has 'flat spots.') In the second formula, the probability that f ( x , y ) ≥ α β ( x ) is therefore equal to 1 -β . Thus, φ β ( x ) comes out as the conditional expectation of the loss associated with x relative to that loss being α β ( x ) or greater.

The key to our approach is a characterization of φ β ( x ) and α β ( x ) in terms of the function F β on X × I R that we now define by

<!-- formula-not-decoded -->

where [ t ] + = t when t &gt; 0 but [ t ] + = 0 when t ≤ 0. The crucial features of F β , under the assumptions made above, are as follows. For background on convexity, which is a key property in optimization that in particular eliminates the possibility of a local minimum being different from a global minimum, see Rockafellar (1970), Shor (1985), for instance.

Theorem 1. As a function of α , F β ( x , α ) is convex and continuously differentiable. The β -CVaR of the loss associated with any x ∈ X can be determined from the formula

<!-- formula-not-decoded -->

In this formula the set consisting of the values of α for which the minimum is attained, namely

<!-- formula-not-decoded -->

is a nonempty, closed, bounded interval (perhaps reducing to a single point), and the β -VaR of the loss is given by

<!-- formula-not-decoded -->

In particular, one always has

<!-- formula-not-decoded -->

Theorem 1 will be proved in the Appendix. Note that for computational purposes one could just as well minimize (1 -β ) F β ( x , α ) as minimize F β ( x , α ). This would avoid dividing the integral by 1 -β and might be better numerically when 1 -β is small.

The power of the formulas in Theorem 1 is apparent because continuously differentiable convex functions are especially easy to minimize numerically. Also revealed is the fact that β -CVaR can

be calculated without first having to calculate the β -VaR on which its definition depends, which would be more complicated. The β -VaR may be obtained instead as a byproduct, but the extra effort that this might entail (in determining the interval A β ( x ) and extracting its left endpoint, if it contains more than one point) can be omitted if β -VaR isn't needed.

Furthermore, the integral in the definition (4) of F β ( x , α ) can be approximated in various ways. For example, this can be done by sampling the probability distribution of y according to its density p ( y ). If the sampling generates a collection of vectors y 1 , y 2 , . . . , y q , then the corresponding approximation to F β ( x , α ) is

<!-- formula-not-decoded -->

The expression ˜ F β ( x , α ) is convex and piecewise linear with respect to α . Although it is not differentiable with respect to α , it can readily be minimized, either by line search techniques or by representation in terms of an elementary linear programming problem.

Other important advantages of viewing VaR and CVaR through the formulas in Theorem 1 are captured in the next theorem.

Theorem 2. Minimizing the β -CVaR of the loss associated with x over all x ∈ X is equivalent to minimizing F β ( x , α ) over all ( x , α ) ∈ X × I R , in the sense that

<!-- formula-not-decoded -->

where moreover a pair ( x ∗ , α ∗ ) achieves the second minimum if and only if x ∗ achieves the first minimum and α ∗ ∈ A β ( x ∗ ) . In particular, therefore, in circumstances where the interval A β ( x ∗ ) reduces to a single point (as is typical), the minimization of F ( x , α ) over ( x , α ) ∈ X × I R produces a pair ( x ∗ , α ∗ ) , not necessarily unique, such that x ∗ minimizes the β -CVaR and α ∗ gives the corresponding β -VaR.

Furthermore, F β ( x , α ) is convex with respect to ( x , α ) , and φ β ( x ) is convex with respect to x , when f ( x , y ) is convex with respect to x , in which case, if the constraints are such that X is a convex set, the joint minimization is an instance of convex programming.

Again, the proof will be furnished in the Appendix. According to Theorem 2, it is not necessary, for the purpose of determining an x that yields minimum β -CVaR, to work directly with the function φ β ( x ), which may be hard to do because of the nature of its definition in terms of the β -VaR value α β ( x ) and the often troublesome mathematical properties of that value. Instead,

one can operate on the far simpler expression F β ( x , α ) with its convexity in the variable α and even, very commonly, with respect to ( x , α ).

The optimization approach supported by Theorem 2 can be combined with ideas for approximating the integral in the definition (4) of F β ( x , α ) such as have already been mentioned. This offers a rich range of possibilities. Convexity of f ( x , y ) with respect to x produces convexity of the approximating expression ˜ F β ( x , α ) in (9), for instance.

The minimization of F β over X × I R falls into the category of stochastic optimization, or more specifically stochastic programming, because of presence of an 'expectation' in the definition of F β ( x , α ). At least for the cases involving convexity, there is a vast literature on solving such problems (Birge and Louveaux (1997), Ermoliev and Wets (1988), Kall and Wallace (1995), Kan and Kibzun (1996), Pflug (1996), Prekopa (1995)). Theorem 2 opens the door to applying that to the minimization of β -CVaR.

## 3 AN APPLICATION TO PORTFOLIO OPTIMIZATION

To illustrate the approach we propose, we consider now the case where the decision vector x represents a portfolio of financial instruments in the sense that x = ( x 1 , . . . , x n ) with x j being the position in instrument j and

<!-- formula-not-decoded -->

Denoting by y j the return on instrument j , we take the random vector to be y = ( y 1 , . . . , y n ). The distribution of y constitutes a joint distribution of the various returns and is independent of x ; it has density p ( y ).

The return on a portfolio x is the sum of the returns on the individual instruments in the portfolio, scaled by the proportions x j . The loss, being the negative of this, is given therefore by

<!-- formula-not-decoded -->

As long as p ( y ) is continuous with respect to y , the cumulative distribution functions for the loss associated with x will itself be continuous; see Kan and Kibzun (1996), Uryasev (1995).

Although VaR and CVaR usually is defined in monetary values, here we define it in percentage returns. We consider the case when there is one to one correspondence between percentage return and monetary values (this may not be true for the portfolios with zero net investment). In this section, we compare the minimum CVaR methodology with the minimum variance approach, therefore, to be consistent we consider the loss in percentage terms.

The performance function on which we focus here in connection with β -VaR and β -CVaR is

<!-- formula-not-decoded -->

It's important to observe that, in this setting, F β ( x , α ) is convex as a function of ( x , α ), not just α . Often it is also differentiable in these variables; see Kan and Kibzun (1996), Uryasev (1995). Such properties set the stage very attractively for implementation of the kinds of computational schemes suggested above.

For a closer look, let µ ( x ) and σ ( x ) denote the mean and variance of the loss associated with portfolio x ; in terms of the mean m and variance V of y we have:

<!-- formula-not-decoded -->

Clearly, µ ( x ) is a linear function of x , whereas σ ( x ) is a quadratic function of x . We impose the requirement that only portfolios that can be expected to return at least a given amount R will be admitted. In other words, we introduce the linear constraint

<!-- formula-not-decoded -->

and take the feasible set of portfolios to be

<!-- formula-not-decoded -->

This set X is convex (in fact 'polyhedral,' due to linearity in all the constraints). The problem of minimizing F β over X × I R is therefore one of convex programming, for the reasons laid out in Theorem 2.

Consider now the kind of approximation of F β obtained by sampling the probability distribution in y , as in (9). A sample set y 1 , y 2 , . . . , y q yields the approximate function

<!-- formula-not-decoded -->

The minimization of ˜ F β over X × I R , in order to get an approximate solution to the minimization of F β over X × I R , can in fact be reduced to convex programming. In terms of auxiliary real variables u k for k = 1 , . . . , r , it is equivalent to minimizing the linear expression

<!-- formula-not-decoded -->

subject to the linear constraints (11), (15), and

<!-- formula-not-decoded -->

Note that the possibility of such reduction to linear programming does not depend on y having a special distribution, such as a normal distribution; it works for nonnormal distributions just as well.

The discussion so far has been directed toward minimizing β -CVaR, or in other words the problem

<!-- formula-not-decoded -->

since that is what is accomplished, on the basis of Theorem 2, when F β is minimized over X × I R . The related problem of finding a portfolio that minimizes β -VaR (Kast et al. (1998), Mauser and Rosen (1999)), i.e., that solves the problem

<!-- formula-not-decoded -->

is not covered directly. Because φ β ( x ) ≥ α β ( x ), however, solutions to (P1) should also be good from the perspective of (P2). According to Theorem 2, the technique of minimizing F β ( x , α ) over X × I R to solve (P1) also does determine the β -VaR of the portfolio x ∗ that minimizes β -CVaR. That is not the same as solving (P2), but anyway it appears that (P1) is a better problem to be solving for risk management than (P2).

In this framework it is useful also to compare (P1) and (P2) with a very popular problem, that of minimizing variance (see Markowitz (1952)):

<!-- formula-not-decoded -->

An attractive mathematical feature of (P3) problem is that it reduces to quadratic programming, but like (P2) it has been questioned for its suitability. Many other approaches could of course also be mentioned. The mean absolute deviation approach in Konno and Yamazaki (1991), the regret optimization approach in Dembo (1995), Dembo and King (1992), and the minimax approach described by Young (1998) are notable in connections with the approximation scheme (17) for CVaR minimization because they also use linear programming algorithms.

These problems can yield, in at least one important case, the same optimal portfolio x ∗ . We establish this fact next and then put it to use in numerical testing.

Proposition. Suppose that the loss associated with each x is normally distributed, as holds when y is normally distributed. If β ≥ 0 . 5 and the constraint (15) is active at solutions to any two of the problems (P1), (P2) and (P3), then the solutions to those two problems are the same; a common portfolio x ∗ is optimal by both criteria.

Proof. Using the MATHEMATICA package analytical capabilities, under the normality assumption, and with β ≥ 0 . 5, we expressed the β -VaR and β -CVaR in terms of mean and variance by

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where exp( z ) denotes the exponential function and erf -1 ( z ) denotes the inverse of the error function

<!-- formula-not-decoded -->

When the constraint (15) is active at optimality, the set X can just as well be replaced in the minimization by the generally smaller set X ′ obtained by substituting the equation µ ( x ) = -R for the inequality µ ( x ) ≤ -R . For x ∈ X ′ , however, we have

<!-- formula-not-decoded -->

where the coefficients c 1 ( β ) and c 2 ( β ) are positive. Minimizing either of these expressions over x ∈ X ′ is evidently the same as minimizing σ ( x ) 2 over x ∈ X ′ . Thus, if the constraint (15) is active in two of the problems, then any portfolio x ∗ that minimizes σ ( x ) over x ∈ X ′ is optimal for those two problems. /diamondmath

This proposition furnishes an opportunity of using quadratic programming solutions to problem (P3) as a benchmark in testing the method of minimizing β -CVaR by the sampling approximations in (17) and their reduction to linear programming. We carry this out in for an example in which an optimal portfolio is to be constructed from three instruments: S&amp;P 500, a portfolio of long-term U.S. government bonds, and a portfolio of small-cap stocks, the returns on these instruments being modeled by a (joint) normal distribution. The calculations were conducted by Carlos Testuri as part of the project in the Stochastic Optimization Course at the University of Florida.

The mean m of monthly returns and the covariance matrix V in this example are given in Table 1 and Table 2, respectively. We took R = 0 . 011 in the constraint (15) on expected loss/return.

First, we solved the quadratic programming problem (P3) for these data elements, obtaining the portfolio x ∗ displayed in Table 3 as the unique optimal portfolio in the Markowitz minimum variance sense. The corresponding variance was σ ( x ∗ ) 2 = 0 . 00378529 and the mean was µ ( x ∗ ) = -0 . 011; thus, the constraint (15) was active in this instance of (P3). Then, for the β -values 0.99, 0.95, and 0.90, we calculated the β -VaR and β -CVaR of this portfolio x ∗ from the formulas in (18) and (19), obtaining the results in Table 4.

With these values at hand for comparison purposes, we proceeded with our approach, based on Theorem 2, of solving the β -CVaR problem (P1) by minimizing F β ( x , α ) over ( x , α ) ∈ X × I R . To approximate the integral in the expression (13) for F β ( x , α ), we sampled the return vector y according to its density p ( y ) in the (multi)normal distribution N ( m , V ) in I R 3 . The samples produced approximations ˜ F β ( x , α ) as in (17). The minimization of ˜ F β ( x , α ) over ( x , α ) ∈ X × I R was converted in each case to a linear programming problem in the manner explained after (17). These approximate calculations yielded estimates x ∗ for the optimal portfolio in (P1) along with corresponding estimates α ∗ for their β -VaR and ˜ F β ( x ∗ , α ∗ ) for their β -CVaR.

The linear programming calculations were carried out using the CPLEX linear programming solver on a 300 MHz Pentium-II machine. In generating the random samples, we worked with two types of 'random' numbers: the pseudo-random sequence of numbers (conventional Monte-Carlo approach) and the Sobol quasi-random sequence (Press et al. (1992), page 310). For similar applications of the quasi-random sequences, see Birge (1995), Boyle et al. (1997), Kreinin et al. (1998). The results for the pseudo-random sequence are shown in Table 5, while those for the quasi-random sequence are shown in Table 6.

In comparing the results in Table 5 for our Minimum CVaR approach with pseudo-random sampling to those that correspond to the optimal portfolio under the Minimum Variance approach in Tables 3 and 4, we see that the CVaR values differ by only few percentage points, depending upon the number of samples, and likewise for the VaR values. However, the convergence of the CVaR estimates in Table 5 to the values in Table 4 (which the Proposition leads us to expect) is slow at best. This slowness might be attributable to the sampling errors in the Monte-Carlo simulations. Besides, at optimality the variance, VaR, and CVaR appear to have low sensitivities to the changes in the portfolio positions.

The results obtained in Table 6 from our Minimum CVaR approach with quasi-random sam-

| Instrument   |   Mean Return |
|--------------|---------------|
| S&P          |     0.010111  |
| Gov Bond     |     0.0043532 |
| Small Cap    |     0.0137058 |

Table 1: Portfolio Mean Return

|           |       S& P |   Gov Bond |   Small Cap |
|-----------|------------|------------|-------------|
| S&P       | 0.00324625 | 0.00022983 |  0.00420395 |
| Gov Bond  | 0.00022983 | 0.00049937 |  0.00019247 |
| Small Cap | 0.00420395 | 0.00019247 |  0.00764097 |

Table 2: Portfolio Covariance Matrix

Table 3: Optimal Portfolio with the Minimum Variance Approach

|      S&P |   Gov Bond |   Small Cap |
|----------|------------|-------------|
| 0.452013 |   0.115573 |    0.432414 |

Table 4: VaR and CVaR obtained with the Minimum Variance Approach

|      |   β = 0 . 90 |   β = 0 . 95 |   β = 0 . 99 |
|------|--------------|--------------|--------------|
| VaR  |     0.067847 |     0.0902   |     0.132128 |
| CVaR |     0.096975 |     0.115908 |     0.152977 |

Table 5: Portfolio, VaR, and CVaR from Min CVaR Approach: Monte Carlo Simulations Generated by Pseudo-Random Numbers ( β value, sample size, three portfolio positions, calculated VaR, deviation from Min Variance VaR, calculated CVaR, deviation from Min Variance CVaR, number of CPLEX iterations, processor time on 300 MHz Pentium II)

|    β |   Smpls # |     S&P |   Gov Bond |   Small Cap |     VaR |   VaR Dif(%) |    CVaR |   CVaR Dif(%) |   Iter |   Time (min) |
|------|-----------|---------|------------|-------------|---------|--------------|---------|---------------|--------|--------------|
| 0.9  |      1000 | 0.3525  |    0.15382 |     0.49368 | 0.06795 |        0.154 | 0.09962 |          2.73 |   1157 |          0   |
| 0.9  |      3000 | 0.55726 |    0.07512 |     0.36762 | 0.06537 |        3.645 | 0.09511 |         -1.92 |    636 |          0   |
| 0.9  |      5000 | 0.42914 |    0.12436 |     0.44649 | 0.06662 |        1.809 | 0.09824 |          1.3  |    860 |          0.1 |
| 0.9  |     10000 | 0.48215 |    0.10399 |     0.41386 | 0.06622 |        2.398 | 0.09503 |         -2    |   2290 |          0.3 |
| 0.9  |     20000 | 0.45951 |    0.11269 |     0.4278  | 0.06629 |       -2.299 | 0.09602 |         -0.98 |   8704 |          1.5 |
| 0.95 |      1000 | 0.53717 |    0.08284 |     0.37999 | 0.09224 |        2.259 | 0.11516 |         -0.64 |    156 |          0   |
| 0.95 |      3000 | 0.54875 |    0.07839 |     0.37286 | 0.09428 |        4.524 | 0.11888 |          2.56 |    652 |          0   |
| 0.95 |      5000 | 0.57986 |    0.06643 |     0.35371 | 0.09175 |        1.715 | 0.11659 |          0.59 |    388 |          0.1 |
| 0.95 |     10000 | 0.47102 |    0.10827 |     0.42072 | 0.08927 |       -1.03  | 0.11467 |         -1    |   1451 |          0.2 |
| 0.95 |     20000 | 0.49038 |    0.10082 |     0.40879 | 0.09136 |        1.284 | 0.11719 |          1.11 |   2643 |          0.7 |
| 0.99 |      1000 | 0.41844 |    0.12848 |     0.45308 | 0.13454 |        1.829 | 0.14513 |         -5.12 |    340 |          0   |
| 0.99 |      3000 | 0.6196  |    0.05116 |     0.32924 | 0.12791 |       -3.187 | 0.14855 |         -2.89 |   1058 |          0   |
| 0.99 |      5000 | 0.63926 |    0.0436  |     0.31714 | 0.13176 |       -0.278 | 0.15122 |         -1.14 |    909 |          0.1 |
| 0.99 |     10000 | 0.45203 |    0.11556 |     0.4324  | 0.12881 |       -2.51  | 0.14791 |         -3.31 |    680 |          0.1 |
| 0.99 |     20000 | 0.45766 |    0.1134  |     0.42894 | 0.13153 |       -0.451 | 0.15334 |          0.24 |   3083 |          0.9 |

Table 6: The Portfolio, VaR, and CVaR from Min CVaR Approach: Simulations Generated by Quasi-Random Sobel Sequences ( β value, sample size, three portfolio positions, calculated VaR, deviation from Min Variance VaR, calculated CVaR, deviation from Min Variance CVaR, number of CPLEX iterations, processor time on 300 MHz Pentium-II)

|    β |   Smpls # |     S&P |   Gov Bond |   Small Cap |     VaR |   VaR Dif(%) |    CVaR |   CVaR Dif(%) |   Iter |   Time (min) |
|------|-----------|---------|------------|-------------|---------|--------------|---------|---------------|--------|--------------|
| 0.9  |      1000 | 0.43709 |    0.12131 |     0.4416  | 0.06914 |         1.9  | 0.09531 |         -1.71 |    429 |          0   |
| 0.9  |      3000 | 0.45425 |    0.11471 |     0.43104 | 0.06762 |        -0.34 | 0.09658 |         -0.41 |    523 |          0   |
| 0.9  |      5000 | 0.44698 |    0.11751 |     0.43551 | 0.06784 |        -0.02 | 0.09664 |         -0.35 |    837 |          0.1 |
| 0.9  |     10000 | 0.45461 |    0.11457 |     0.43081 | 0.06806 |         0.32 | 0.09695 |         -0.02 |   1900 |          0.3 |
| 0.9  |     20000 | 0.46076 |    0.11221 |     0.42703 | 0.0679  |         0.08 | 0.09692 |         -0.06 |   4818 |          0.6 |
| 0.95 |      1000 | 0.43881 |    0.12065 |     0.44054 | 0.09001 |        -0.21 | 0.11249 |         -2.95 |    978 |          0   |
| 0.95 |      3000 | 0.43881 |    0.12065 |     0.44054 | 0.09001 |        -0.21 | 0.11511 |         -0.69 |    407 |          0   |
| 0.95 |      5000 | 0.46084 |    0.11218 |     0.42698 | 0.09036 |         0.18 | 0.11516 |         -0.64 |    570 |          0.1 |
| 0.95 |     10000 | 0.45723 |    0.11357 |     0.4292  | 0.09016 |        -0.05 | 0.11577 |         -0.12 |   1345 |          0.2 |
| 0.95 |     20000 | 0.45489 |    0.11447 |     0.43064 | 0.09023 |         0.03 | 0.11577 |         -0.12 |   1851 |          0.7 |
| 0.99 |      1000 | 0.52255 |    0.08846 |     0.38899 | 0.1249  |        -5.47 | 0.14048 |         -8.17 |    998 |          0   |
| 0.99 |      3000 | 0.4303  |    0.12392 |     0.44578 | 0.12801 |        -3.12 | 0.15085 |         -1.39 |    419 |          0   |
| 0.99 |      5000 | 0.45462 |    0.11457 |     0.43081 | 0.13073 |        -1.06 | 0.14999 |         -1.95 |    676 |          0.1 |
| 0.99 |     10000 | 0.39156 |    0.13881 |     0.46963 | 0.13288 |         0.57 | 0.15208 |         -0.59 |   1065 |          0.2 |
| 0.99 |     20000 | 0.46065 |    0.11225 |     0.4271  | 0.13198 |        -0.11 | 0.15211 |         -0.57 |   1317 |          0.5 |

pling exhibit different and better behavior. There is relatively fast convergence to the values for the Minimum Variance problem. When the sample size is above 10000, the differences in CVaR and VaR obtained with the Minimum CVaR and the Minimum Variance approaches are less than 1%.

## 4 AN APPLICATION TO HEDGING

As a further illustration of our approach, we consider next an example where a NIKKEI portfolio is hedged. This problem, out of Mauser and Rosen (1999), was provided to us by the research group of Algorithmics Inc. Mauser and Rosen (1999) considered two ways of hedging: parametric and simulation VaR techniques. In each case, the best hedge is calculated by one-instrument minimization of VaR, i.e., by keeping all but one of the positions in the portfolio fixed and varying that one, within a specified range, until the VaR of the portfolio appears to be as low as possible. Here, we show first that when the same procedure is followed but in terms of minimizing CVaR, the one-instrument hedges obtained are very close to the ones obtained in terms of minimizing VaR. We go on show, however, that CVaR minimization has the advantage of being practical beyond the one-instrument setting. Positions of several, or even many, instruments may be adjusted simultaneously in a broader mode of hedging.

As in the application to portfolio optimization in the preceding section, the calculations could be reduced to linear programming by the kind of maneuver described after (16), which adds an extra variable for each scenario that is introduced. This would likely be advantageous for hedges involving the simultaneous adjustment of positions in a large number of instruments (say &gt; 1000). But we demonstrate here that for hedges with relatively few instruments being adjusted, nonsmooth optimization techniques can compete with linear programming. In such techniques there is no need to add extra variables, and the dimension of the problem stays the same regardless of how many scenarios are considered.

Table ?? shows a portfolio that implements a butterfly spread on the NIKKEI index, as of July 1, 1997. In addition to common shares of Komatsu and Mitsubishi, the portfolio includes several European call and put options on these equities. This portfolio makes extensive use of options to achieve the desired payoff profile. Figure ?? (reproduced from Mauser and Rosen (1999)) shows the distribution of one-day losses over a set of 1,000 Monte Carlo scenarios. It indicates that the normal distribution fits the data poorly. Therefore, Minimum CVaR and Minimum Variance approaches could, for this case, lead to quite different optimal solutions.

Table 7: NIKKEI Portfolio, reproduced from Mauser and Rosen (1999).

| Instrument            | Type   | Day to Maturity   | Strike Price (10 3 JPY)   |   Position (10 3 ) | Value (10 3 JPY)   |
|-----------------------|--------|-------------------|---------------------------|--------------------|--------------------|
| Mitsubishi EC 6mo 860 | Call   | 184               | 860                       |               11.5 | 563,340            |
| Mitsubishi Corp       | Equity | n/a               | n/a                       |                2   | 1,720,00           |
| Mitsubishi Cjul29 800 | Call   | 7                 | 800                       |              -16   | -967,280           |
| Mitsubishi Csep30 836 | Call   | 70                | 836                       |                8   | 382,070            |
| Mitsubishi Psep30 800 | Put    | 70                | 800                       |               40   | 2,418,012          |
| Komatsu Ltd           | Equity | n/a               | n/a                       |                2.5 | 2,100,000          |
| Komatsu Cjul29 900    | Call   | 7                 | 900                       |              -28   | -11,593            |
| Komatsu Cjun2 670     | Call   | 316               | 670                       |               22.5 | 5,150,461          |
| Komatsu Cjun2 760     | Call   | 316               | 760                       |                7.5 | 1,020,110          |
| Komatsu Paug31 760    | Put    | 40                | 760                       |              -10   | -68,919            |
| Komatsu Paug31 830    | Put    | 40                | 830                       |               10   | 187,167            |

For the 11 instruments in question, let x be the vector of positions in the portfolio to be determined, in contrast to z , the vector of initial positions in Table ?? (the fifth column). These vectors belong to I R 11 . In the hedging, we were only concerned, of course, with varying some of the positions in x away from those in z , but we wanted to test out different combinations. This can be thought of in terms of selecting an index set J within { 1 , 2 , . . . , 11 } to indicate the instruments that are open to adjustment. In the case of one-instrument hedging, for instance, we took J to specify a single instrument but consecutively went through different choices of that instrument.

Having selected a particular J , for the case when J contains more than one index, we imposed, on the coordinates x j of x , the constraints

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

but on the other hand thus taking

Figure 1: Distribution of losses for the NIKKEI portfolio with best normal approximation, (1,000 scenarios), reproduced from Mauser and Rosen (1999).

The constraints (21) could be used of course to eliminate the variables x j for j / ∈ J from the problem, which we did in practice, but this formulation simplifies the notation and facilitates comparisons between different choices of J . The absolute values appear in (20) because short positions are represented by negative numbers.

Let m be the vector of initial prices (per unit) of the instruments in question and let y be the random vector of prices one day later. The loss to be dealt with in this context is the initial value of the entire portfolio minus its value one day later, namely

<!-- formula-not-decoded -->

The corresponding function in our CVaR minimization approach is therefore

<!-- formula-not-decoded -->

The problem to be solved, in accordance with Theorem 2, is that of minimizing F β ( x , α ) over X × I R . This is the minimization of a convex function over a convex set.

To approximate the integral we generated sample points y 1 , y 2 , . . . , y q and accordingly replaced F β ( x , α ) by

<!-- formula-not-decoded -->

an expression that is again convex in ( x , α ), moreover piecewise linear. Passing thereby to the minimization of ˜ F β ( x , α ) over X × I R , we could have converted the calculations to linear programming, but instead, as already explained, took the route of nonsmooth optimization. This involved working with the subgradient (or generalized gradient) set associated with ˜ F β at ( x , α ), which consists of all vectors in I R 11 × I R of the form

<!-- formula-not-decoded -->

We used the Variable Metric Algorithm developed for nonsmooth optimization problems in Uryasev (1991), taking β = 0 . 95, which made the initial β -VaR and β -CVaR values of the portfolio be 657,816 and 2,022,060.

The results for the one-instrument hedging tests, where we followed our approach to minimize β -CVaR with J = { 1 } , then with J = { 2 } , and so forth, are presented in Table ?? . The optimal hedges we obtained are close to the ones that were obtained in Mauser and Rosen (1999) by minimizing β -VaR. Because J was comprised of a single index, x was just one-dimensional in these tests; minimization with respect to ( x , α ) was therefore two-dimensional. The algorithm needed less than 100 iterations to find 6 correct digits in the performance function and variables.

For testing purposes, we employed the MATHEMATICA version of the variable metric code on a Pentium II, 450MHz machine. (The FORTRAN and MATHEMATICA versions of the code are available at http://www.ise.ufl.edu/uryasev). The constraints were accounted for by nonsmooth penalty functions. Each run took less than one minute of computer time. The calculation time could be significantly improved using the algorithm implemented with FORTRAN or C, however such computational studies were beyond the scope of this paper.

After finishing with the one-instrument tests, we tried hedging with respect the last 4 of the 11 instruments, simultaneously. The optimal hedge we determined in this way is indicated in Table ?? . The optimization did not change the positions of Komatsu Cjun2 670 and Komatsu Paug31 760, but the positions of Komatsu Cjun2 760 and Komatsu Paug31 830 changed not only in magnitude but in sign. In comparison with one-instrument hedging, we observe that the multiple instrument hedging considerably improved the VaR and CVaR. In this case, the final β -VaR equals -1,400,000 and the final β -CVaR equals 37,334.6, which is lower than best one-dimension hedge with β -VaR=-1,200,000 and β -CVaR=363,556 (see line 9 in Table ?? ). Six correct digits in the performance function and the positions were obtained after 400-800 iterations of the variable metric algorithm in Uryasev (1991), depending upon the initial parameters. It took about 4-8

minutes with MATHEMATICA version of the variable metric code on a Pentium II, 450MHz.

In contrast to the application in the preceding section, where we used linear programming techniques, the dimension of the nonsmooth optimization problem does not change with increase in the number of scenarios. This may give some computational advantages for problems with a very large number of scenarios.

This example clearly shows, by the way, the superiority of CVaR over VaR in capturing risk. Portfolios are displayed that have positive β -CVaR but negative β -VaR for the same level of β = 0 . 95. The portfolio corresponding to the first line of Table ?? , for instance, has β -VaR equal to -205,927 but β -CVaR equal to 1,183,040. A negative loss is of course a gain 1 . The portfolio in question will thus result with probability 0.95 in a gain of 205,927 or more. That figure does not reveal, however, how serious the outcome might be the rest of the time. The CVaR figure says in fact that, in the cases where the gain of at least 205,927 is not realized, there is, on the average, a loss of 1,183,040.

## 5 CONCLUDING REMARKS

The paper considered a new approach for simultaneous calculation of VaR and optimization of CVaR for a broad class of problems. We showed that CVaR can be efficiently minimized using Linear Programming and Nonsmooth Optimization techniques. Although, formally, the method minimizes only CVaR, it also lowers VaR because CVaR ≥ VaR.

We demonstrated with two examples that the approach provides valid results. These examples have relatively low dimensions and are offered here for illustrative purposes. Numerical experiments have been conducted for larger problems, but those results will be presented elsewhere in a comparison of numerical aspects of various Linear Programming techniques for portfolio optimization.

There is room for much improvement and refinement of the suggested approach. For instance, the assumption that there is a joint density of instrument returns can be relaxed. Furthermore, extensions can be made to optimization problems with Value-at-Risk constraints. Linear Programming and Nonsmooth Optimization algorithms that utilize the special structure of the Minimum CVaR approach can be developed. Additional research needs to be conducted on various theoretical and numerical aspects of the methodology.

1 VaR may be negative because it is defined relative to zero, but not relative to the mean as in VaR based on the standard deviation.

Table 8: Best Hedge, Corresponding VaR and CVaR with Minimum CVaR Approach: One-Instrument Hedges ( β = 0 . 95).

| Instrument            | Best Hedge   | VaR        | CVaR      |
|-----------------------|--------------|------------|-----------|
| Mitsubishi EC 6mo 860 | 7,337.53     | -205,927   | 1,183,040 |
| Mitsubishi Corp       | -926.073     | -1,180,000 | 551,892   |
| Mitsubishi Cjul29 800 | -18,978.6    | -1,170,000 | 553,696   |
| Mitsubishi Csep30 836 | 4381.22      | -1,150,000 | 549,022   |
| Mitsubishi Psep30 800 | 43,637.1     | -1,150,000 | 542,168   |
| Komatsu Ltd           | -196.167     | -1,180,000 | 551,892   |
| Komatsu Cjul29 900    | -124,939     | -1,200,000 | 593,078   |
| Komatsu Cjun2 670     | 19,964.9     | -1,220,000 | 385,698   |
| Komatsu Cjun2 760     | 4,745.20     | -1,200,000 | 363,556   |
| Komatsu Paug31 760    | 3,1426.3     | -1,120,000 | 538,662   |
| Komatsu Paug31 830    | 19,356.3     | -1,150,000 | 536,500   |

Table 9: Initial Positions and Best Hedge with Minimum CVaR Approach: Simultaneous Optimization with respect to Four Instruments ( β = 0 . 95; VaR of best hedge equals -1,400,000, whereas CVaR Equals 37334.6.

| Instrument         | Position in Portfolio   | Best Hedge   |
|--------------------|-------------------------|--------------|
| Komatsu Cjun2 670  | 22,500                  | 22,500       |
| Komatsu Cjun2 760  | 7,500                   | -527         |
| Komatsu Paug31 760 | -10,000                 | -10,000      |
| Komatsu Paug31 830 | 10,000                  | -10,000      |

## ACKNOWLEDGMENTS

Authors are grateful to Carlos Testuri who conducted numerical experiments for the example on comparison of the Minimum CVaR and the Minimum Variance Approaches for the Portfolio Optimization. Also, we want to thank the research group of Algorithmics Inc. for the fruitful discussions and providing data for conducting numerical experiments with the NIKKEI portfolio of options.

## REFERENCES

Andersson, F. and Uryasev, S. (1999). Credit Risk Optimization With Conditional Value-AtRisk Criterion. Research Report 99-9. ISE Dept., University of Florida, August.

Artzner, P., Delbaen F., Eber, J.M., and Heath, D. (1997). Thinking Coherently. Risk, 10, November, 68-71.

Artzner, P., Delbaen F., Eber, J.M., and Heath, D. (1999). Coherent Measures of Risk. Mathematical Finance , 9, 203-228.

Birge, J.R. (1995). Quasi-Monte Carlo Methods Approaches to Option Pricing. Technical report 94-19. , Department of Industrial and Operations Engineering, The University of Michigan, 15p.

Birge,J.R. and Louveaux, F. (1997). Introduction to Stochastic Programming , Springer Verlag, pp. 448.

Boyle, P.P., Broadie, M., and Glasserman, P. (1997). Monte Carlo Methods for Security Pricing Journal of Economic Dynamics and Control. Vol. 21 (8-9), 1267-1321.

Bucay, N. and Rosen, D. (1999). Credit Risk of an International Bond Portfolio: a Case Study. ALGO Research Quarterly. Vol.2, 1, 9-29.

Dembo, R.S. (1995). Optimal Portfolio Replication. Algorithmics Technical paper series. 95-01.

Dembo, R.S. and King, A.J. (1992). Tracking Models and the Optimal Regret Distribution in Asset Allocation. Applied Stochastic Models and Data Analysis. Vol. 8, 151-157.

Duffie, D. and Pan, J. (1997). An Overview of Value-at-Risk. Journal of Derivatives. 4, 7-49.

Embrechts, P. (1999). Extreme Value Theory as a Risk Management Tool. North American Actuarial Journal , 3(2), April 1999.

Embrechts, P., Kluppelberg, S., and Mikosch, T. (1997). Extremal Events in Finance and Insurance. Springer Verlag.

Ermoliev, Yu. and Wets, R. J-B. (Eds.) (1988). Numerical Techniques for Stochastic Optimization , Springer Series in Computational Mathematics, 10.

Kall, P., and Wallace, S.W. (1995). Stochastic Programming , John Wiley &amp; Sons, pp. 320.

Kast, R., Luciano, E., and Peccati, L. (1998). VaR and Optimization. 2nd International Workshop on Preferences and Decisions, Trento, July 1-3 1998.

Kan, Y.S., and Kibzun, A.I. (1996). Stochastic Programming Problems with Probability and Quantile Functions , John Wiley &amp; Sons, pp. 316.

Konno, H. and Yamazaki, H. (1991). Mean Absolute Deviation Portfolio Optimization Model and Its Application to Tokyo Stock Market. Management Science. 37 , 519-531.

Kreinin, A., Merkoulovitch, L., Rosen, D., and Michael, Z. (1998). Measuring Portfolio Risk Using Quasi Monte Carlo Methods. ALGO Research Quarterly. Vol.1, 1, 17-25.

Litterman, R. (1997a). Hot Spots and Hedges (I). Risk . 10 (3), 42-45.

Litterman, R. (1997b). Hot Spots and Hedges (II). Risk , 10 (5), 38-42.

Lucas, A., and Klaassen, P. (1998). Extreme Returns, Downside Risk, and Optimal Asset Allocation. Journal of Portfolio Management , Vol. 25, No. 1, 71-79.

McKay, R. and Keefer, T.E. (1996). VaR Is a Dangerous Technique. Corporate Finance Searching for Systems Integration Supplement. Sep., pp. 30.

Markowitz, H.M. (1952). Portfolio Selection. Journal of finance. Vol.7, 1, 77-91.

Mausser, H. and Rosen, D. (1999). Beyond VaR: From Measuring Risk to Managing Risk. ALGO Research Quarterly. Vol.1, 2, 5-20.

Pflug, G.Ch. (1996). Optimization of Stochastic Models : The Interface Between Simulation and Optimization. Kluwer Academic Publishers, Dordrecht, Boston.

Pflug, G.Ch. (2000). Some Remarks on the Value-at-Risk and the Conditional Value-at-Risk. In.'Probabilistic Constrained Optimization: Methodology and Applications', Ed. S. Uryasev, Kluwer Academic Publishers, 2000

Prekopa, A. (1995). Stochastic Programming. Kluwer Academic Publishers, Dordrecht, Boston.

Press, W.H., Teukolsky, S.A, Vetterling, W.T., and Flannery, B.P. (1992). Numerical Recipes in C. Cambridge University Press.

Pritsker, M. (1997). Evaluating Value at Risk Methodologies, Journal of Financial Services Research , 12:2/3, 201-242.

RiskMetrics TM . (1996). Technical Document , 4-th Edition, J.P.Morgan, December 1996.

Rockafellar, R.T. (1970). Convex Analysis. Princeton Mathematics, Vol. 28, Princeton Univ. Press.

Shapiro, A. and Wardi, Y. (1994). Nondifferentiability of the Steady-State Function in Discrete Event Dynamic Systems. IEEE Transactions on Automatic Control , Vol. 39, 1707-1711.

Shor, N.Z. (1985). Minimization Methods for Non-Differentiable Functions. Springer-Verlag.

Simons, K. (1996). Value-at-Risk New Approaches to Risk Management. New England Economic Review , Sept/Oct, 3-13.

Stambaugh, F. (1996). Risk and Value-at-Risk. European Management Journal, Vol. 14, No. 6, 612-621.

Stublo Beder, T. (1995). VAR: Seductive but Dangerous. Financial Analysts Journal , Sep.Oct., 12-24.

Uryasev, S. (1991). New Variable-Metric Algorithms for Nondifferential Optimization Problems. J. of Optim. Theory and Applic. Vol. 71, No. 2, 359-388.

Uryasev, S. (1995). Derivatives of Probability Functions and Some Applications. Annals of Operations Research , V56, 287-311.

Uryasev, S. (2000). Conditional Value-at-Risk: Optimization Algorithms and Applications. Financial Engineering News , 14, February 2000.

Young, M.R. (1998). A Minimax Portfolio Selection Rule with Linear Programming Solution. Management Science. Vol.44, No. 5, 673-683.

Zenios, S.A. (Ed.) (1996). Financial Optimization , Cambridge Univ. Pr.

Ziemba, W.T. and Mulvey, J.M. (Eds.) (1998): Worldwide Asset and Liability Modeling , Cambridge Univ. Pr.

## Appendix

Central to establishing Theorems 1 and 2 is the following fact about the behavior with respect to α of the integral expression in the definition (4) of F β ( x , α ). We rely here on our assumption that Ψ( x , α ) is continuous with respect to α , which is equivalent to knowing that, regardless of the choice of x , the set of y with f ( x , y ) = α has probability zero, i.e.,

<!-- formula-not-decoded -->

Lemma. With x fixed, let G ( α ) = ∫ y ∈ I R m g ( α, y ) p ( y ) d y , where g ( α, y ) = [ f ( x , y ) -α ] + . Then G is a convex, continuously differentiable function with derivative

<!-- formula-not-decoded -->

Proof . This lemma follows from Proposition 2.1 in Shapiro and Wardi (1994).

Proof of Theorem 1 . In view of the defining formula for F β ( x , α ) in (4), it is immediate from the Lemma that F β ( x , α ) is convex and continuously differentiable with derivative

<!-- formula-not-decoded -->

Therefore, the values of α that furnish the minimum of F β ( x , α ), i.e., the ones comprising the set A β ( x ) in (6), are precisely those for which Ψ( x , α ) -β = 0. They form a nonempty closed interval, inasmuch as Ψ( x , α ) is continuous and nondecreasing in α with limit 1 as α →∞ and limit 0 as α →-∞ . This further yields the validity of the β -VaR formula in (7). In particular, then, we have

<!-- formula-not-decoded -->

But the integral here equals

<!-- formula-not-decoded -->

where the first integral on the right is by definition (1 -β ) φ β ( x ) and the second is 1 -Ψ( x , α β ( x )) by virtue of (27). Moreover Ψ( x , α β ( x )) = β . Thus,

<!-- formula-not-decoded -->

This confirms for β -CVaR formula in (5) and finishes the proof of Theorem 1.

/diamondmath

Proof of Theorem 2 . The initial claims, surrounding (10), are elementary consequences of the formula for φ β ( x ) in Theorem 1 and the fact that the minimization of F β ( x , α ) with respect to ( x , α ) ∈ X × I R can be carried out by first minimizing over α ∈ I R for fixed x and then minimizing the result over x ∈ X .

Justification of the convexity claim starts with the observation that F β ( x , α ) is convex with respect to ( x , α ) whenever the integrand [ f ( x , y ) -α ] + in the formula (4) for F β ( x , α ) is itself convex with respect to ( x , α ). For each y , this integrand is the composition of the function ( x , α ) ↦→ f ( x , y ) -α with the nondecreasing convex function t ↦→ [ t ] + , so by the rules in Rockafellar (1970) (Theorem 5.1) it is convex as long as the function ( x , α ) ↦→ f ( x , y ) -α is convex. The latter is true when f ( x , y ) is convex with respect to x . The convexity of the function φ β ( x ) follows from the fact that minimizing of an extended-real-valued convex function of two vector variables (with infinity representing constraints) with respect to one of these variables, results in a convex function of the remaining variable (Rockafellar (1970), pp. 38-39). /diamondmath