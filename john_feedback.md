Dear Pranjal congratulations again on your defense yesterday. However it is clear from Matt's email that there are basic steps you need to
    complete on submitting a finalized, properly formatted thesis to meet the requirements for graduation. I finally found an email from
    March 2nd that you sent with a link to the thesis on Google drive, but there were 5 attachments and they were not combined in the
    format of a thesis and I had expected to receive that prior to the defense. Meanwhile it appears you made significant updates to some
    of the chapters such as the one you presented yesterday so I will need to see the finalized, properly formatted thesis before I can sign
    off on it. According to Matt you have until April 20th but to give me time to read/review please get it to me ASAP.

    Overall the chapter on the review of RL is well written and very valuable, but the version of the paper you presented yesterday,
    while technically impressive, was full of jargon and terms such as "softplus" "infoNCE", "embeddings" "FashionCLIP" that are 
    not defined and presume the reader has a level of familiarity with a narrow technical/computer science oriented literature that forces most
    ordinary economists like myself to search the internet to find out definitions/explanations of terms that could have been put into
    this chapter. I don't want to overly delay you for meeting the submission deadline to graduate but I know you are capable of writing
    more clearly for non-specialist readers and explaining the terminology and the thesis gives you the space to do this. 

    So I will expect an update of the chapter that also fixes typos or other questions that might come up after a closer reading, such    
    as the point yesterday where your notation allowed for the two type mixture probabilities \pi to be arbitary functions of the d vector
    but in actuality you just estimated a single pi, 1-pi value for the proportions of the two types.  I will not raise quarrels with technical
    issues such as getting "correct standard errors" and inferential/identification aspects, but as was commented yesterday in your defense
    the way you wrote down the model suggests the possibility for non-identification and if you don't have an identified structural model
    the question arises about how much we can trust counterfactuals that come out of it. So when it comes to submitting this to a journal
    (which I strongly encourage you to do) this will have to be carefully addressed.

    But these larger questions are not ones I expect you to solve before the 20th but should definitely be noted/hedged in the chapter and then after
    the thesis is approved (and I sign off) I can send a more focused critique of what referees are likely to say if you were to submit this to a journal
    so that you can improve this and get it published in a top journal. I think the chapter is very promising and Marketing Science could be
    a potentially good destination that would be probably a big benefit for your career even if you do not anticipate going into academia in the
    future. I think a good model of writing style that you might benefit from is the way Chengjun Zhang wrote his thesis, which has some 
    similarities do what you are doing except that Chengjun had fixed set of attributes of cell phones and did not need to extract these
    attributes from photographs of cell phones, and also he had less info on demographics but the actual search clickstream.  But for example,
    a very insightful graph Chengjun did was to illustrate the cross elasticities for different consumers and how a purchaser of a cheap
    flip-phone has cross price elasticities that are high only for other cheap flip-phones but very small for the high end smart phones, 
    whereas the reverse is true for consumers who search for and end up buying the high end smart phones.  I wonder if you can produce
    a similar graphic of the cross price elasticities from your model showing that for older women the cross price elasticities are high for the
    types of dresses they prefer, and are low for dresses that young women choose and vice versa, so that the model is in effect finding
    consumer segments and patterns of shopping for dresses in different segments.

    Again, these will be part of comments I provide (which are entirely *optional* for you in terms of re-writing the thesis chapters
    to try to get them published) and which I DON'T EXPECT TO BE DONE AS PART OF THE ACTUAL THESIS YOU HAND IN. I want to
    be pragmatic and help you meet the thesis deadline and here my main requirements are 1) fulfill the Georgetown formatting guidelines,
    2) minimize typos and mistakes to extent possible, and 3) make sure the thesis chapters are readable/understandable to non-specialist
    readers so jargon, abbreviations, and other technical terminology is defined/explained and the chapters end with a more or less
    intuitive summary of what has been learned from your work.

    I think you are capable of this and am happy to talk and meet if there is anything unclear in the above. Again, I think your work is
    impressive, but it was done really quickly and perhaps without as much attention as you could have devoted to readability and
    intuition  (with the exception of the RL chapter, which does very well on that score). I am not asking for or expecting huge changes
    in what you already have done, but if you can start sending me chapters as they are completed that will go into the final thesis,
    I can start to read them right away and provide any comments so you have enough time to fix any problems/typos I might find
    before the 20th.

    sincerely John


    ok, have you defined or provided a paragraph on what InfoNCE is?  This does have a very intuitive
    explanation as a logit model estimation with embeddings being "latent charasteristic vectors"
and so explaining this to readers makes the chapter more interesting and accessible to people
who do not already know what InfoNCE is.  there are identification issues involved in InfoNCE
and if the vectors are entirely latent, what is the identifying "normalization" for these models?
Some of these issues/problems are discussed in this paper
https://arxiv.org/abs/2407.00143

But given the Monday deadline, as long as you have some short discussion/explanation of
key technical terms and avoid too many mnemonics or terms such as "three tower" unless
you explain what they are at the time you first use them, I think that would be enough.  But
for a paper to go to say Marketing Science, you probably want to say more about the methodology
and the history of InfoNCE and other methods. But this is not something to worry about for Monday's
deadline: just go through and try to minimize the number of technical terms you introduce that have
no explanation in your thesis and require someone to do a Google search to understand what you 
are talking about.

Also using terms such as "contrastive loss" when the objective function can also be described
as a negative log likelihood function is again coming across as speaking to an entirely different
audience than the one I presume you are trying to write for. It is fine to use the synonyms used
in the ML/NN literature, but only after you have noted their relation to the more traditional/established
notions of these things in terms of statistical inference. 


In equation (2.2) for example you hvae not defined what C(j^+) is. Presumably it is the set of *non-purchased items*
and if so, the formula seems wrong as it does not include exp(\hat y_{ij+}) in the denominator to ensure the CCPs
add up to 1.  YOu do say on page 16

When
this denominator covers every other product in the category, Equation (2.2) is algebraically
identical to the log-likelihood of a standard multinomial choice model with utility u⊤
i vj /τ
and no price term.

but with the changes you have, this appears rather ad hoc, and there is no attempt at motivation for why doing
what you are proposing here as the first stage of this process is a reasonable/justifiable thing to  do. I am not
requiring you to do this for the thesis, to be clear, but I think if you were to send this to an economics or marketing
journal without providing that justification, it would be one of the things reviewers might jump on, looking for
an excuse to reject your paper. So the more you can justify and provide intuition for what you do and not create
an impresssion that what you are doing has a large ad hoc element to it, you will have a much better chance of
getting an R&R at one of the top journals if you submit this (which I hope you do and encourage you to do, though
it will require some patience and doing some major rewrites to better explain what you did and why it makes sense).

There are also questions about the methodology and if the "fine tuning" is creating an endogeneity problem or other
econometric issues due to the presence of estimation noise in the first stage of a 2 stage estimation procedure. An alternative
"one step" estimator is to use the standard InfoNCE embeddings "out of the box" without this first stage of processing and then use deep nets
for consumer demographics d and the product embeddings (without any price info included, but allowing text in addition to
images) performs relative the approach you have chosen.  You could then allow price to enter nonlinearly in the estimator
instead of as you do in (2.4). This method is closer to the approach of Farrell, Liang and Misra (2021)
and would be a natural benchmark for comparison, and an extension of what they did since now you are using deep nets to uncover "charateristics"
of products and show the heterogeneity in consumer evaluation of/reaction to these characteristics.  Your approach does this
too, but you don't explain whether your approach was done for computational expediency or because this first stage results
in a better model fit, or both.  I expect that this will be the part of your paper that most of the questions/critiques will be raised
so it is a good idea to anticipate them, raise them yourself by discussing other ways of doing this and explaining why you 
chose to do it the way you did, convincing the reader that your way is better. But these are all suggestions for things to consider
*AFTER* you hand in your thesis on Monday.

Again to get the thesis in by Monday just focus on trying to put in short explanations of technical terms that an
average economist may not know and hopefully this will not take too long and I would assume this will be
easy and not jeopardize your ability to meet the Monday
deadline. Let me know if you don't think it is feasible to do that, and we can talk about some compromise. 

sincerely John 

Dear Pranjal these are all good responses but remember my critiques are in the spirit of "devil's advocate"
        to prompt you to justify/explain what you do, but only *after* handing in the thesis on Monday.
        I am sure there are good reasons for most of your choices and I am not asking you to take the
        time now to explain/defend them to me. And even after it is handed in, of course any suggestions
        I might make are not ones you might agree with or feel that are worth doing and some of them
        may represent misunderstanding of what you are doing.  So there will be plenty of time to deal
        with all of that. For now just deal with the 'factual' parts of fixing any unreferenced equations
        and other objects, and trying to define technical terms to make the chapter more readable to
        a non-specialist in ML/NN.

    sincerely John 



Dear Pranjal the substantive area where I think your chapter 2 will
receive pushback is on the
     fine tuning and particularly using price encoded as a 64
dimensional vector in the "3 tower"
     model leading to the "contrastive" "affinity" in equation (1). This
looks like a standard logit
     utility function and price is entering into the "embeddings" that
are subsequently used
     in the choice model (4).  I find this hard to explain/understand
except that it was done out
     of computational convenience/necessity, but I think it would be
better to use the 512 dimensional
     embeddings as the v inputs to a deep net where there is a term
alpha_c(v,d) for the utility of
     product v to demographic d and a price sensitivity coefficient
alpha_p(d) that would be natural
     to assume is independent of v though you could try including it.
then Price enters only where
     "it should" but in your specification price is implicitly affecting
your 64 dimensional embeddings
     via the affinity definitions, so both (v,d) from your approach
implicitly depend on p.

     this makes your approach seemingly less useful for designing new
products v since it seems
     to require advance knowledge of p to determine what v might be, so
this might be regarded
     as a form of "endogeneity" inherent in your approach. If you have a
good answer to this then
     the paper should definitely explain it, but it would also seem like
referees might ask you
     to compare the peformance of deep net specification that only uses
the 512 dimensional embeddings
     straight from FashionCLIP without the "fine tuning" as inputputs to
a deep or shallow net.
     After all the 512 dimensional input layer could produce a smaller
dimensional output layer
     such as the 64 dimensions you are using, but instead of a two stage
estimation process it would
     be all done in one stage. So I am not convinced that there is such
a big computational saving
     or performance gain in what you are doing, but of course you did
the work and know it well and
     I could just be wrong here, but I think you need to explain better
the method you chose since
     I could imagine referees asking questions similar to the one I am
raising here.

     You do have a discussion of these issues on page 12, but it does
not really seem that convincing
     to me for some reason and when you mention *price contamination*
there, I can only think that
     (1) is indeed introducing price contamination/endogeneity and the
approach is sort of distorting
     embeddings by using choice data to better explain choice data, but
in that process you are getting
     embeddings that look more like utility function coefficients for
the "affinities" (1) than "objective"
     attributes of dresses. So from the perspective of new product
design, your approach would seem to
     require re-estimation of the affinities since the v seems to
require knowledge of the price p that
     would be charged for it.

     Another unclear aspect is that figure 3 seems to indicate that only
product images are used but
     footnote 4 indicates that actually you averaged the 512 dimensional
image and text embedding.
     It would be interesting to see whether the text descriptions
contain independent information
     from the images but this approach seems to preclude that. For
example the dress material, cotton,
     polyester, wool, etc may not be easy to see from the image but
would be contained presumably in
     the text and that might be very important info for some purposes.

     It would also be nice if your model could provide more
     insights into things such as if there are some attributes such as
color that are not that
     important because for a given style there are a range of colors
that can be chosen and something
     approximating one's "favorite color" are available for all dresses?
Similarly for
     dress size: size would not be an important attribute unless there
were "stockouts" and some sizes
     not available.   So the more fundamental "attributes" would seem to
be 1) fabric, and 2) dress style,
     with color and size being secondary if we assume that for online
purchases there are no stockouts
     and once one picks a fabric/style product, they have the choice of
a set of colors and sizes and these
     second stage choices are pretty much the same for all dresses. Here
I could be wrong, and perhaps
     many/most dresses are not available in a wide color palette and
     some ladies shop lexicographically based on color then size, style
and lastly fabric and others
     have other lexicographic orderings of their preferences for
dresses. But it was not clear from your
     chapter how "dress size" gets recorded (via measurements of
breast/waist, length? or small, medium large?)
     and if that is part of the "choice set" or you assume that the
right size is always available and abstract
     from it in your choice model.  From table 1 it seems that color is
recorded, and figure 14 shows you
     do have fabric from text descriptions, but given the difference in
marginal costs it would be interesting
     to know if your model does capture "fabric" in the v vector, or if
it might imply that H&M could
     swap a high cost fabric such as denim or satin for a lot cost one
(e.g. cotton or polyester) and the
     model v would not change much, suggesting that H&M should not be
selling much of the high cost
     fabrics in its dresses. So an "experiment" of the simple act of
swapping a high cost fabric for a low
     cost one in the same dress style, size and color and seeing how the
model predicts demand would
     change would seem like a good test of your specification, or
whether adding a "fabric dummy" to your
     utility specification (5) would help improve fit, which might
suggest that somehow the embedding
     strategy of combining image and text embeddings into a single
averaged vector is not adequately
     capturing the information context that is contained in the image
and text separately (e.g. in the 512 dimensional
     fashionCLIP embeddings of image and text description)

   Sincerely John

--
John Rust
Emeritus Professor
Georgetown University
Department of Economics
583 Intercultural Center
Washington, DC 20057-1036
http://editorialexpress.com/jrust


Pranjal <pp712@georgetown.edu>
Attachments
Tue, Apr 21, 8:52 PM (12 days ago)
to John

Dear Prof, 

The way I see it is I have two options: 

a) Two step estimation. Split the sample, estimate embeddings on A and then the demand-system on B. This is perfectly valid because the first stage freezes the embeddings as fixed covariates. This is the standard strategy called sample-splitting, and avoids the generated regressors problem. We have a clean two stage approach: the first stage can be loose and wild and ML-cowboy style, and the second stage needs to be principled. The drawback is that a) we throw half the sample out and b) we are unable to get good estimates for new users and products which were not there in the old sample. I think a lot of ML+Econometrics work operates in this domain. I also chose this for simplicity. In our problem, we do not have a good demographic vector and the first stage does a good job of creating them. It also can compress the item embeddings so we reduce parameter count. 

b) One step estimation. This involves jointly estimating embeddings and the core model. I'm looking into this. Embeddings are not basically random vectors. And one needs to do either full Bayesian inference or MAP estimation or variational inference. This is a bit computationally demanding but there are some papers that do this-- I cite them below. The advantage would be that a) we have one clean model with no confusion, b) lower uncertainty due to more data and c) embeddings for cold start (users/items who are new). I think marketing journals have Bayesian models so it will be acceptable. 

  Compiani, Morozov & Seiler (2025) -- They fix product embeddings
   from a pre-trained CLIP/text encoder and estimate consumer heterogeneity from market-level choices in one
   pass via BLP contraction + GMM. The product side is frozen, I go further by estimating        
  individual d_i rather than a population distribution.                                                    
                                                                                                           
  Donnelly, Loaiza-Maya & Frazier (2021) -- shows that individual consumer latent vectors and product latent
   vectors can be recovered jointly from purchase data via EM + variational inference, at scale (333K      
  products). The bilinear price term gamma_i^T lambda_j is the closest prior art to your alpha_i =
  softplus(alpha_0 + a^T d_i).     

Which direction would you recommend? 

I attach a quick note made with claude. 
 One attachment
  •  Scanned by Gmail

Pranjal <pp712@georgetown.edu>
Wed, Apr 22, 3:53 AM (11 days ago)
to John

Dear Prof 

I’m going to work a bit on my visa issues now, I need to apply for an EAD card and I’m running a bit late on that. Once I’m done I can get back to this.

-Pranjal

John Rust
Wed, Apr 22, 11:00 AM (11 days ago)
to me

Daer Pranjal by all means, you do not need to feel compelled to respond immediately
    to my emails. They are intended to be helpful advice but not a continuation of some
    kind of extended PhD defense.

    Here is another way to express my question/concern about your two stage approach
    to estimating the choice model for dresses. As I see it the "affinity model" for fine tuning
    the 64 dimensional embeddings in equation (1) is already itself a coherent discrete choice
    model of dresses that uses the 562 dimensional embeddings from FashionCLIP, i.e. the

512-dimensional

embeddings from product photographs and product descriptions using a pre-trained vision-

language model, alongside 50 dimensions of categorical attributes (e.g., color group, garment

type).


    the 64x1 vector v_j could then be interpreted as a "vector of characteristics of dress j"
    and the embedding d_i could be interpreted as "the consumer's preferred characteristics for a dress".
    With the normalization of these vectors to the unit circle in R^{64} as an identification normalization
    then (1) looks like an ordinary logit model with a specific utility function being the inner product
    of d_i with v_j less a price term given by the softplus coefficient.  Why isn't this model already enough
    as a model of consumer choice?  The inner product then relates to cosine similarity between the
    consumer's ideal product attributes and the actual product attributes and produces where this inner
    product (or cosine) is 1 are the most preferred, those where it is -1 are the least preferred, etc.

    Once you have done this, you are using these vectors as "fine tuned" embeddings in R^64 to estimate
    the discrete choice model again but it is not clear how much "value added" this 2nd stage is providing
    or why it needs to be done since the first stage already has the interpretation of discrete choice model
    albeit with a pretty specific utility function. 

    The fact that you can estimate (1) belies the statement on page 12 that

A 64-number vector per product is small enough to load, index, and pass through
subsequent networks at reasonable speed and memory cost; the 512-dimensional CLIP output
multiplied across roughly 39 969 products and every consumer-item pair in the panel would be
slow and expensive to work with.

    But (1) does involve an estimation of deep nets with actually 562 dimensional inputs, right?
    Also for a big business like H&M, the rationalization that using the ful CLIP output would be
    slow probably would not cut it: it seems more like you are rationalizing a desire to estimate 
    the model on your laptop at home rather than go out and get the extra processing power you
    might need on the cloud. So probably this rationalization might not cut it for referees at a journal
    either unless you can show that this dimensionality reduction results in very little loss of information
    and a massive speedup in estimation.

    Instead I think (1) just suggests you could estimate richer and less restrictive deep or shallow nets directly using
    the FashionCLIP output and the 50 dimensional binary attributes such as garment type (i.e. fabric dummies?)
    and a vector of indicators for consumer types that is more flexible/general than specification (1) and this
    would be a "one stage" estimation that should be feasible if you are already able to estimate (1) itself.

    The more flexible neural net specification of the form alpha(v,d) for the utility and a price coefficient
    like -exp(alpha_p(d,p)) (assuming you want to enforce downward sloping demand) would seem to
    result in a better fit than the existing specification (1) and would be a natural alternative to compare
    it to.

    I don't see why Bayesian estimation is necessary here. It can be done by Maximum likelihood. Why
    introduce a prior? Especially if you want to compare to existing well used methods, I think Bayesian
    estimation make the different methods/approaches less easy to  since Bayesian will be prior-dependent.
    Also unless price endogeneity is a real problem, BLP may not be a good thing to compare to either since
    it requires good estimates of market shares. But with so many dress products, many market shares could
    be zero or imprecisely estimated. I think maximum likelihood  with product-specific dummies
    would be a better choice since these dummies can capture the "product specific characteristics" that
    are the omitted \xi_j terms in the BLP analysis which throws away completely the micro data to use
    only market share. Why would you want to do that?  With product dummies interacted with demographic
    info d, you can get a richer model than BLP allows which implicitly requires all consumers to evaluate
    the unobserved attributes \xi_j the same way. With maximum likelihood of a model with product dummies
    you can allow d to interact, ii.e. implicitly for \xi_j to be a function of d and differ across cosumer types.

    then you can have a clean comparison: doing maximum likelihood, one for a more traditional model
    but with Farrell, Misra et. al. heterogeneity in the logit coefficients but the coefficients are also
    functions of product or class specific dummies, and your model where you avoid product or class dummies
    and instead use the FashionClIP embeddings v_j to capture product attributes along with other
    info such as fabric dummes that might not be captured well by v_j

    The latter point might we worth discussing more too: my understanding of contrastive loss training
    of FashionCLIP is that it tries to best match text descriptions to images. But if images cannot convey
    fabric type well, then the v_j from FashionCLIP probably cannot pick up Fabric type well: the fabric
    type is in the text description but the CL training will treat that info as orthogonal to the goal of
    best matching text to image photo. So that's why separately extacting that other sort of info and
    not just relying on machine learning to do everything is a good idea. It is where your knowledge as
    a human and training as an economist can have a lot of value added compared to a completely
    robotic/mechanical approach that does not attempt to rely on any human intuition. 

    So as you suggest in your own chapter on RL, the big gains can come from both: using the best
    of ML but recognizing some of its limitations and using human understanding to augment where
    ML is likely to underperform and finally, using human understanding to provide an interpretation
    of the findings and not just treat the model as a choice predictor. Focusing the article on the new
    insights we get on how different women choose dresses and what features influence them and 
    converting these into intuitive "stories" readers can relate to and make sense will do a lot to convince
    readers that you are introducing a valuable new tool that both predicts well and is practical because
    it captures essential insights/understandings into how women shop for dresses and choose which
    ones to buy, and hopefully, to predict well new dress designs that H&M might introduce in the future.

    I would recommend trying to tone down use of jargon too such as introducing "softplus" with no
    explanation/definition  of what it is, while not explaining why softplus is better than simply
    using the exponential function if you are interested in enforcing a sign restriction on price. 
    But more than that, why is it necessary to enforce a sign restriction?  I think it would be valuable
    if you could allow for more nonlinear, unrestricted specifications where price enters such as
    alpha(d,p) where neither sign nor linearity is enforced. Presumably if the model is a good one and
    women do prefer lower prices to higher ones, the estimation should be "discovering" this and it
    should not need to be imposed a priori. But of course sometimes things don't work out that way
    and if you do get too many counterintuitive price effects, it is worth discussing in the paper rather
    than just imposing a restriction that in effect sweeps the problems under the rug. 

    if there are enough markdowns, the question is whether this constitutes "quasi random price experimentation"
    that would be sufficient variation to get good estimates of price. If mark downs occur more for
    products that are "duds" then indeed there might be an issue with endogeneity and more thought
    is necessary on how to deal with this. but a "regression discontinuity" strategy that looks at sales
    before and after a markdown might still show that demand is downward sloping, and thus conditioning
    on rich enough product dummies seems intuitively like a way to do something that approximates
    a RDD design and gets accurate price effects without having to do BLP and think about relevant instrumental
    variables. So another aspect of your analysis to focus on is also whether it generates reasonable own and
    cross price elasticities compared to the more traditional approaches. 

    But coming back to the original point, I don't still understand well what the "second stage" is gaining
    you after doing the first stage estimation of (1) which is in effect already a discrete choice model
    using the FashionCLIP output.  If you can provide a good answer to that, I think it will go a long way
    to addressing questions referees might have about your 2 stage approach as well. Perhaps I am making
    too big of a deal about this and if Nate and Sanjog don't seem confused or concerned, maybe it is just
    me.  But the main thing your "latent class deep logit" model is adding that is not in (1) seems to be
    the multiple types, but it seems you could extend estimation of (1) to a two type model as well. 

    By the way, the introduction of equation (5) on page 16 is confusing since right after you write
that equation you refer to x_j, the "64-dimensional
three-tower item vector for the five non-Dress categories in Section C, and the 576-dimensional
concatenation of that same three-tower vector with the 512-dimensional raw FashionCLIP
vector for the Dress master (Section 5 footnote)."  But there is no x_j in equation (5). Instead the
reader has to go down to equation (8) on page 16 to see that t_j^c is a function of x_j.  It would be
a good idea for you to go back over your write up of the model since you are probably too familiar
with everything by now and are not reading it from the reader's perspective, but instead presuming
readers are as familiar with all the terminology and notation as you are.  Focusing on making the paper
easier to read/understand and more intuitive, avoiding too much jargon without prior explanation/definition
will do a lot to make the paper more accessible and help people better understand your contribution.

best regards, John 

Pranjal <pp712@georgetown.edu>
Attachments
Wed, Apr 22, 2:38 PM (11 days ago)
to John

Agree with you on: 
- BLP is not the right tool 
- Penalized MLE might be enough in this case (no need to try variational inference). I do want to avoid Bayesian estimation unless necessary. 
- Yes, probably need to de-jargonize even further and further fixes to the notation. 

What I used ultimately in the paper so far is the 64D item embedding + 512 CLIP in the second stage, along with the 64D user embedding. I found this gave best out of sample Mcfadden R2 (about 0.08) and reasonable price elasticities and IIA departures. So your point is well taken that 512 CLIP can be powerful on its own, and there is a case for joint estimation. The two stage model worked like this: estimate one global model on 12 categories and that took 2 days, and once done --- you can estimate separate category level latent class deep logit models. And since they are on separate samples strictly by time, we avoid the generated regressors problem. This is also nice because the embeddings are used in other analysis: hedonics, event studies. Two step approaches thus carry a lot of convenience---and that explains its popularity. 

So I think it is possible to use 512 CLIP for the items and this is sufficiently rich. The main issue is that we do not have any user information beyond age, location; and so the bottleneck is really user embeddings---and we need them for the "radically deep heterogeneity" (beauty lies in the eyes of the beholder". One thing that works is a pooled estimation of many categories of items (joint likelihood over many category logit models), so the user embeddings are "common" to all but the item embeddings are different across categories. This forces a single user embedding to explain many models. I am trying out "one stage only" approaches, where we essentially learn the user embeddings (which is what is missing) while the item embedding is fixed. 

So the one-step estimation model, that I am trying out should cover all the suggestions you made: please see attached a 1 pager. 
 One attachment
  •  Scanned by Gmail

Pranjal <pp712@georgetown.edu>
Wed, Apr 22, 11:09 PM (11 days ago)
to John

I'm getting some promising results when I retrain the model using a joint-embedding-and-latent-class InfoNCE model that can be used to create both embeddings as well as do demand estimation. Its a big heavy, so my laptop keeps crashing, but I will apply some bag of computational tricks to get it to work. Hopefully by tomorrow, we should see good results. 

John Rust
Fri, Apr 24, 3:21 PM (9 days ago)
to me

Dear Pranjal: sorry for the delay in responding but I spend all of yesterday on campus adivising
a number of other students including Satyam who obtained an online new service data set with
clickstream data that might become his job market paper, and Zefan, Mingao and Jeremy all
had questions and wanted some of my help/advice. I also had a meeting this morning with
a team of coauthors on my other papers on the auto market in Denmark. So much interesting
stuff to be doing, with so little time to do it?

I want to make clear that I think your project is really interesting and promising and I don't
want to come across as telling you what to do, and encourage you to seek out advice from
other experts and if Nate, Sanjog and others give you advice/comment, it could be a lot different
than what I am saying and ultimately different people will have different concerns and you cannot
make everyone perfectly happy, but the most important person it to make yourself happy and do
the work the way you think it should be done.

I read the 1 pager but not sure if I follow it completely. Is "latent class" a "latent type" of consumer?
the mixture probabilities \pi_{ik} aren't they the type probabiliies? If so, given the comments at your defense that
making them functions of observables when the type coefficients are also rich functions of consumer
types leads to identification problems, I thought you would be just estimating unconditional type probabilities
as you actually did in the thesis, so it would be just 1 probability given the 2 type model. Are you now
proposing to estimate this richer specification but if so, how do you address the identification concerns?

I will just refrain from now from further comments on the 1 pager since it will be too terse for me to
understand what you are doing and just lead to more confusion and slow you down. Just let me see the
next version when you have it and I can comment on it then.

I do recommend some graphs to show to an average reader whether FashionCLIP generates accurrate
product groupings. As I understand it, FashionCLIP chooses weights to produce 512 dimensional
latent vectors to lie on the unit circle that maximizes a likelihood of matching the photo of a dress
to its text description,  so these vectors should be similar for dresses in the same "dress category" and
different for dresses in diifferent dress categories. The normalization makes the use of correlation
coefficient (cosine similarity, but again I prefer terms that an average economist understands/relates to)
is a good measure  of "similarity", so you can you produce a plot showing that products in the same
product category, e.g. "prom dresses" or "wedding dresses" or "cocktail party dresses" have latent
vector correlations that are closer to 1 and for pairs of dresses in different categories, these correlations
are closer to 0 or even potentially negative?   Another exercise would be to use k-means clustering of
these vectors based on their correlations (or L2 norm) as a measure of distance: does k-means clustering
product categories that match the ones that H&M uses on its website?

Some intuitive graphical means of showing that the FashionCLIP embeddings do capture essential
features that people see in looking at dresses would be helpful in convincing them that this is the
right way to proceed with this analysis.

But the alternative is to just estimate a model with "dress dummies" that can be consumer-type specific.
The question is whether this type of model, more standard in the literature, would fit better than a model
that uses "deeper attributes" of dresses and would facilities potentially better predictions of demand for
completely new product designs.  The dress dummies are the standard way to capture unobserved attributes
of dresses that can lead to price endogeneity and the question is whether an approach that uses FashionCLIP
embeddings instead effectively controls for these unobserved attributes equally well to generate similar
price coefficients and price elasticities, cross elasticities, etc. 

On the question of product groupings and consideration sets, I think it could be justified to think
of consumers choosing dresses from a narrow class of dress type (e.g. prom dresses) and not the set
of all dresses, combined with a no purchase outside good alternative. Comparing the results when the
choice set is all dresses compared to a model with narrower "consideration sets" (equal to the class of
dresses the consumer clicks on when buying) would be interesting and potentially you might get different
and better results from models with more constrained choice sets, especially in terms of price sensitivity
and cross price elasticities.  

I understand that 70% of your data are online purchases. If you have observations of people
who are online but did not purchase (i.e. chose the outside good) then I would recommend restricting
to online only purchases since limited inventory is less likely to be a problem in such purchases,
whereas for in-store purchases, many sizes/styles might not be in stock.

but I am not clear if your data is choice based, i.e.you only see cases of people who bought, not
people who arrived at a store or online, shopped but did not buy.  If you don't have those latter
observations of choice of the "outside good" I think a discussion is in order, because they you are
estimating a discrete choice model of which dress is purchased conditional on at least one is purchased,
but a full model of pricing needs to capture the probability that a person "arriving" at the H&M website
searches but does not purchase a dress and how elastic that binary probability is. this is related
to the problem in BLP of determining "market size" and how to define what the "outside good" is. 

If you have addressed these issues already in the paper and I just missed them, let me know and I will
re-read. 

Also on your 1 pager, I don't think EM algorithm is necessary: the Adam optimizer should be
able to handle direct optimization of the mixed likelihood function and I don't know why you have
the penalty term on the log-likeihoood in the M step. For the Who is More Bayesian? paper, I never
needed to use the EM algorithm and got perfectly good results much faster by direct optimization 
of the mixture likelihood by standard quasi-newton and trust region methods. though you have many
more parameters here, it is not clear to me why you don't just use Adam to maximize a mixture
likelihood directly without any EM algotithm and without any parameter penalty function.

AGain., don't feel obligated to reply to this. These are just my off the cuff reactions. I can go back
and do a closer reading once you have new results you want me to look at and I can also comment
on exposition so that it will be easier for non-specialists to read your paper. I think you could do yourself
a disservice if you use too much jargon and introduce too many methods/algorithms without adequate
explanation for why they are used that the overall proceess seems too complicated and unintuitive.
I think a better way is to relate this to work that readers are already familiar with, showing it is 
a do-able addition using already developed Fashion-CLIP embeddings to estimate a model that captures
deeper and better measures of "attributes" of dresses, while also using other features in the data 
(such as fabric) that may not stand out clearly in the Fashion-CLIP embeddings, so you are using a 
best of both worlds approach to develop a model that can successfully predict the sales of new
dresses, something H&M is doing constantly in its business to stay ahead of ever changing fashion trends.

best regards, John 

Pranjal <pp712@georgetown.edu>
Fri, Apr 24, 4:21 PM (9 days ago)
to John

I think the feedback you have given is quite sufficient at this stage. 
Yes, while I state the general form I typically only estimate unconditional type/class probabilities. 
I'm also directly optimizing the mixture loss. 
The way I'm treating non-availability of "did not purchase" is to first estimate choice conditional on purchase and then calibrate an outside option's mean utility such that when we solve for market shares we get something that gives H&M about 5% inside share. 
In the paper, I do have graphs to show that FashionCLIP is able to cluster the products very well. Visually, it is picking up distinct styles. This is verified via K-means as well as other clustering algorithms. 
Yes, we are restricting online-only purchases. 
So I feel I'm covering most of the bases here. I am obtaining positive results by doing a joint InfoNCE + latent class logit for all 12 categories at once such that we have one common set of consumer embeddings and common alpha (price) and beta (aesthetics) networks. By using much more data, and having common weights, we are able to get quite good out of sample R2. I will share some results with you shortly. 

John Rust
Attachments
Fri, Apr 24, 5:08 PM (9 days ago)
to me

Dear Pranjal Ok, good luck.  I will focus on advising Hoang on his thesis and
    how to revise it for submission. When you want more feedback, feel free to contact me.
    Also I am happy to send a version to Gunter for his feedback.

    I attach his disseration submission that won the Bass Award at Marketing Science in 2007.
    Others have won it have gone to very distinguished careers such as K. Sudhir

    https://www.informs.org/Recognizing-Excellence/Community-Prizes/Marketing-Science-Society/Frank-M.-Bass-Dissertation-Paper-Award

    and it you position your paper in the right way, I think your contribution could be submitted
    for this award too and would be happy to nominate it.

    I attached Gunter's paper in case you want to look at examples of the style of paper that wins
    such award. I think it starts with a good practical question, and does not just introduce "technique
    for technique's sake" but instead cleverly introduces new methodology that illustrates an important
    question of interest and ends up providing insight into the original practical question.

    Having a dynamic model is not important for winning the award, if you can
    show how deep net technology can be used to successfully predict the demand respond to 
    alternative new products. It would be really cool if you could go back and show example of 
    "duds" that H&M introduced and quickly discontinued for lack of sales, vs successful products
    that were introduced and continued and sold well. If you could show your model was able to predict
    that the duds would be duds and the successful new dresses would be successful, that would be
    a great intuitive and easy to understand outcome of the methdology.  Of course it is asking too much
    to expect that even very sophisticated new technology will not lead to prediction errors, but if there
    is a way to show that you have developed a tool that could be more successful than a more trial and
    error approach run by human experts or some algorithm that H&M might be using, that could potentially
    be an award winning paper. But again it is still asking a lot, because you do not have access to all the
    fashion trends that H&M is following when it is probably copying successful dress designs elsewhere
    and producing them more cheaply in its Asian factories. But you don't really have to: if H&M is a fashion
    follower and gets an idea of a dress to copy from wherever, as long as you see the new dresses it is introducing
    and see the sales responses, if your model can be more successful compared to other standard approaches in
   predicting demand I think it could be a submission that not only can get published in Marketing Science
    but could be a contender for the Bass Award. 

    But of course there are other ways to motivate what you are doing in practical terms and other
    ways to illustrate the model if the suggestion above does not grab you or seem feasible to carry out
    with your data.

    best regards, John 
