# Transcript of Youtube

0:00
speculations on test time scaling say what you will about open AI they make iconic graphs this graph is from the
0:07
gpt3 paper it's one of several different graphs that demonstrate as language model parameters get larger the models
0:15
perform better at zero shot tasks this graph and many others like it has
0:20
occupied the interest of the language model Community for The Last 5 Years it really changed the way we think about
0:26
many different problems and how we build design scale and invest in models in
0:31
fact right now it's safe to say that there are nuclear power plants being built just to support this graph
0:37
recently OpenAI released a new graph given the impact of their previous ones
0:42
I take this pretty seriously in this graph on the left hand side we see a curve
0:47
that looks pretty similar to The Curve we've seen before in this curve we see that more training time compute leads to
0:54
consistently better accuracy on a hard task on the right hand side of this graph we see new curve this curve looks
1:02
similar in that we're seeing compute contrasted with the performance of the model what's different is that this
1:08
curve is showing test time compute and what we are seeing is the performance on this task get much better as we add more
1:14
test time compute to the system this is new we haven't seen this before in language modelling and it's a topic of
1:20
great interest before we dive into this topic more it's worth noting what these problems actually look like well they're
1:27
pretty hard we're not just doing ret Ral or things that look a bit more like pattern matching we're doing full-on
1:34
reasoning in technical mathematical problems one essay that often comes up when discussing these sort of scaling
1:40
challenges is the bitter lesson you've probably read this blog post before but it's worth giving you a sense of what
1:46
it's saying and how it applies to this problem the bitter lesson is based on the historical observations that AI
1:52
researchers have often tried to build knowledge into their agents this always helps in the short term and is personally satisfying to the researcher
1:59
but in the long run it plateaus and even inhibits further progress and breakthrough progress eventually arrives
2:05
by opposing approaches based on scaling computation by search and learning I
2:10
bring it up today because it's been a popular conversation among researchers in this space what we've seen for the
2:16
last 5 years is increase in the learning capability of models what we might be seeing now is a move towards search in
2:23
particular a type of search that is facilitated by learning to allow us to scale on some of these more technical
2:29
problems in preparing this talk I also watched some of the recent Talks by noam brown
2:34
one of the contributors to this system one thing he recommended when talking about his past work is he said that the
2:40
most important lesson is that I and other researchers simply didn't know how much of a difference scaling up search would make if I had seen these scaling
2:47
results at the start of my PhD I would have shifted to a researching search algorithms for poker much sooner we
2:52
probably would have gotten super hum poker Bots much sooner as well while he's talking here about game playing the
2:58
goal of his conversation position is to encourage the field in general to think more about search one of the other
3:04
papers he brings up is a paper I found to be very influential around people in this area even though it's not more
3:10
widely known this is a paper from 2021 that discusses scaling laws for board
3:15
games in this setting it's easy to move back and forth between more training and
3:20
more test time search and in the paper they describe how these two terms relate
3:26
to each other in particular in these curves here they show that there is trade-off that with more training time
3:32
in the model itself you can learn better systems but that this alone doesn't replace the test time compute by
3:39
plotting these two points together you can understand the relative relationships between these two and learn how to actually apply them in
3:45
practice when talking about similar techniques for language models people often bring up a paper from openai from
3:52
2021 in this paper they're going to train what we'll call a learned verifier
3:57
to do this they will use a generative model model to produce hundreds of different solutions they'll then have
4:03
experts look at these Solutions and select which are right and which are wrong using this information they can
4:09
then train a verifier this verifier will tell you if you're doing well on the problem and can be used at test time to
4:16
try to improve your answers while there are a lot of details to the paper itself
4:21
one of the most important results is they show that searching against this learned verifier can lead to
4:27
improvements even upon just training on the actual good answers themselves in
4:32
this graph here the Orange Line shows the accuracy of the model that is running against the verifier compared to
4:38
a model that's just trained on the trajectories themselves this is an argument for moving beyond the standard
4:44
supervised fine-tuning or sft more to A system that utilizes a learned verifier
4:51
in order to inject new signal into the model this allows you to utilize that verifier to improve the model at test
4:57
time as we'll see we don't think this is exactly what open AI is doing but it
5:02
gives you a sense of how they were exploring early uses of test time compute in developing their systems and
5:09
actually on that note we really know very little about what is going on in open AI in fact I'm not going to really
5:16
focus too much on trying to predict what they actually are doing but I'd like to use the release of this model as an
5:22
opportunity to provide a survey about what's going on in the open research field in this talk I'll give a survey of
5:29
the public literature related to opening eyes 01 as part of this process I also
5:34
took the opportunity to call up a lot of different researchers in the field it's a nice perk of being a professor that a
5:40
lot of different people will talk to me and for this talk I talked to about 25 different people about what they think
5:45
is going on finally I will include some rumours from social media I'm not going to wait these too highly because who
5:52
knows what's going on in practice uh but that will give us some constraints for thinking about what the system could be
5:57
doing the Talk itself will have four more parts first we'll talk about some of the clues behind what o1 might be
6:03
then I'll provide some technical background behind the techniques that are actually used in these systems
6:09
finally I'll discuss four different suspects why I think they're interesting what papers have been written about them
6:15
and whether or not they might be what's going on finally I'll end by talking about some implications for researchers
6:21
or open- Source practitioners in this area to gather Clues let's start with
6:26
open ai's own words in a blog post with the release of 01 they wrote a lot about
6:32
this topic but much of it is mostly just marketing there are two sentences though
6:37
that give us a sense about what might be happening first they say our large scale
6:43
reinforcement learning algorithm teaches the model how to think productively using its Chain of Thought in a highly
6:50
data efficient training process this sentence gives us three Clues into what might be happening the first is that the
6:56
system is using reinforcement learning the precise definition of reinforcement learning has become quite hard to pin
7:03
down and it means different things to different people I'm going to adapt the most loose definition and just say it
7:10
requires some signal from some sort of verifiable problem we're going to assume
7:15
we don't have supervised data and we need to acquire the signal in other ways secondly the method uses Chain of
7:22
Thought specifically it's using Chain of Thought as its method of increasing test
7:27
time compute what this means is that we're not doing any sort of search during test time in fact the system is
7:35
just generating a very long output stream and using that to make its final
7:40
prediction we'll discuss this more throughout the talk finally the system is data efficient what this means is
7:46
that it's learned from a relatively small set of data examples this is not making any Claim about compute
7:53
efficiency or even parameter efficiency just that the amount of actual problems it needs is Rel relatively small where
8:01
relative here is compared to say training on the entire internet in addition to this sentence there are
8:06
several other assumptions that people seem to be making about these models the first is that it is a single final
8:12
language model that generates an extremely long and coherent Chain of Thought just to make that clear once
8:18
again it's just a model that babbles to itself until it thinks it has good
8:24
enough information to make a guess at the answer to your hard problem second we assume the is not following from
8:31
expert examples this is not to say there isn't a huge amount of human supervision
8:36
it's just that that supervision is not given in the form of direct human answers to questions it's not copying
8:43
along with what humans did finally there's an assumption that the behaviors
8:48
of exhibits are learned that means they come somehow from data or self-play but
8:54
not from being given to the model explicitly of these assumptions the most important one is this idea of Chain of
9:01
Thought let's describe what this means informally and then we'll dive into a bit more of a formal explanation later
9:07
in The Talk The informal definition is that the model is going to generate intermediate steps in the process of
9:13
producing an answer these intermediate steps are not supervised but are simply sampled from the language model as we go
9:20
so in this example on the right we are given a question written in red and then we produce four steps of Chain of
9:26
Thought written in green below each of these steps compute some intermediate term in the process finally we produce a
9:33
final answer at the end which is used to evaluate our performance on the problem in the same blog post open AI
9:41
highlights this use of Chain of Thought They say 01 learns to hone its Chain of Thought and refine the strategies it
9:47
uses it learns to recognize and correct its mistakes it learns to break down tricky steps into simpler ones it learns
9:54
to try a different approach when the current one isn't working it's hard to pull too much signal out of this
10:00
sentence but it really highlights that Chain of Thought is where the action is happening unlike other systems that
10:07
build in complex search as part of their test time this model is simply utilizing
10:12
the Chain of Thought to do these steps as it goes in their blog post open AI additionally included some examples of
10:19
the Chain of Thought for the system you're not actually able to see this Chain of Thought in the model they
10:24
released but we can look at some of the chains they provided first off we can see that Chain of Thought for a
10:30
programming problem just to note again this is something the model itself produced in the process of solving the
10:36
problem what we can see is that the model has produced an outline of all the steps it would like to produce outline
10:43
is numbered and includes complex sub-steps if you read the rest of the Chain of Thought you can see that it's
10:49
following this outline in the process of producing its answer in another example
10:54
there's a form of rudimentary planning we can see that the system is aware of the time constraints it needs to answer
11:01
the problem it also is able to stop and propose different options and choose which of these it would like to follow
11:07
while this is all in English it's using cues like first or option one in order
11:13
to specify the intermediate steps another ability that we see in these chains is forms of backtracking so in
11:19
this example for a math problem it describes some intermediate term that it might need to compute it then stops in
11:26
the middle and says well actually this may not help us directly this allows the
11:31
model to go back and determine that it might want to say something different again this looks a bit like search but
11:37
it's not actually being performed with traditional search it's simply the model talking to itself in order to determine
11:43
the answer final ability we see is something like self-evaluation here we see it say let's
11:50
analyze each option it then specifies the options it might want to consider
11:55
and it asks itself is that a good explanation the answer is a bit informal it says H and then goes on to the next
12:02
option itself but again this is an ability that can be used by the model in
12:08
order to explore different possibilities and determine which ones might make sense so in summary Chain of Thought
12:13
here is providing our method of test time scaling while the actual words in the Chain of Thought look like search
12:20
and planning in a classical sense this is not actually being run at test time for the model how does it learn to do
12:26
this well this is the big mystery they claim that reinforcement learning is needed to induce this Behavior the rest
12:33
of the talk I want to explore how this might actually come about and look at some of the papers in the literature
12:38
which talk about how you can get models to learn to do something like this such that you can scale their test time
12:44
compute to get there though we're going to need some technical background so for this section we're just going to focus
12:50
on formalizing this idea of Chain of Thought we're not going to do any learning but simply talk about what it
12:57
means to start from a question go through say four or five steps of intermediate reasoning and then come up
13:03
with an answer formally we'll assume our problem specification is called X this
13:09
is just the question being asked that we need to solve our final solution will be called y this will represent say the
13:17
conclusion or answer to our math problem in between we'll look to produce a
13:22
series of steps Z1 through ZT these are not individual words but instead full
13:28
step steps on the way to produce our final answer we'll abstract a bit away from the fact that this is a language
13:34
model and just think about it producing steps along in this chain the final goal
13:40
at the bottom of this slide is to produce a distribution over answers y conditioned on our input X this
13:46
distribution is defined by taking an expectation over these latent Chain of Thought steps
13:52
Z as a warm-up let's consider how standard Chain of Thought is done we
13:57
can't actually compute the expectation over all possible intermediate steps so instead we run ancestral sampling this
14:04
is a fancy term but just means let the language model generate until it produces an answer specifically we'll
14:11
sample steps Z these are represented by the green dots in the little picture
14:16
until we get to An Answer y represented by the dot on the right we can think
14:22
about the amount of test time compute being applied here as T where that represents each one of the intermediate
14:28
steps on the way to our answer in general I use these graphs on the right to demonstrate how the Chain of Thought
14:34
is being used in these processes many papers have noted that there is a way to get better answers to these problems
14:42
instead of taking a single chain of thought and using it to produce the answer we can instead sample n chain of
14:48
thoughts once we have these end different chain of thoughts we can take a majority vote in order to determine
14:54
the majority answer in this diagram here each one of these chain of thoughts thoughts is sampled independently and
15:01
then we do some sort of normalization the answers that are most common are the ones we decide this provides a strong
15:08
Baseline and a way to utilize more test time compute to slightly improve our answers you can obviously do this to a
15:15
large extent but people have found that it doesn't lead to some of the amazing results that we're seeing in the 01 blog
15:20
post the second piece of Machinery we need is a verifier explicitly we'll
15:25
assume that we have an automatic verifier but we only have it at training time we'll Define this automatic
15:32
verifier as taking some answer why and telling us if it is wrong or right this
15:38
verifier might be say regular Expressions to check that we've solved a math problem or it might be something
15:43
more complex like full-on unit tests for code again just to make it clear we don't have this at test time but we are
15:51
going to utilize it as a way to provide training signal for us to produce a better model throughout this talk I'm
15:58
going to assume that we have an automatic verifier this is a common assumption in much of the research and I
16:04
think it's a reasonable assumption for solving these problems that being said it's not clear whether open AI is
16:10
actually utilizing automatic verifiers or whether they're using learned verifiers in some of their papers they
16:16
explicitly try to learn verifiers for some technical problems their argument is that this can produce more general
16:23
purpose models and it's a way for them to utilize their large annotation facilities in order to improve prove
16:29
their models in the case of learned verifiers there are some interesting research challenges for instance One
16:35
Challenge is that with a learned verifier if the generator produces say crazy Solutions sometimes the Learned
16:41
verifier gets confused and accepts them in this graph on the right they show that for a math problem the model will
16:48
continue getting better but then it will Plateau and even get worse as they take more samples they discuss how this is a
16:54
challenge with a learned verifier and I have to assume they've collected a lot of data and thought about this problem a lot more in recent years however since
17:02
we're poor both in gpus and in data annotations we'll focus instead more on the automatic verifier situation once
17:09
you have a verifier there are many new things you can do one idea is to do rejection sampling this is a way of
17:16
getting some distribution of chain of thoughts that yield correct answers we're going to do this just by sampling
17:23
again and different chains and simply keeping the ones that are verified in in
17:29
my notation on the right on the top picture we see all the different samples that we picked and then on the bottom
17:34
picture we see the chain of thoughts that led to correctly verified Solutions I'll use this little square box on the
17:41
right to indicate which Solutions were successfully verified this process may be extremely compute intensive but it gives
17:48
us a way to get some of the good chain of thoughts that lead to the verified Solutions we're going to assume for now
17:54
that you can plausibly get some solutions using this procedure although it's it's not obvious for hard problems
18:00
that this will ever yield a correct answer the other thing we can do is to apply the same process but starting from
18:07
an intermediate step in our chain so in this example here we've run three steps of Chain of Thought and from there we'll
18:14
do what's called a roll out this roll out is the same as rejection sampling but it just starts from the intermediate
18:20
place in our process this can be used to tell us how good we are doing from any
18:25
step in the chain it's not guaranteed to always work for good problems or bad problems but at least gives us a
18:31
direction in which to move in our system given this formal background we now come to our main goal we would like to learn
18:38
a model that can take into account these latent chain of thoughts we can write this down explicitly as a maximum
18:45
likelihood problem where we're interested in learning the model that performs as well as it can at producing
18:50
verified Solutions we can then think about this as marginalizing out over all possible chain of thoughts that lead to
18:57
the correct answer of course this problem is combinator extremely difficult we can't with even nearly
19:04
infinite compute really do this marginalization there are many different possible steps at each point and when we
19:10
start talking about chains of thousands of steps long this becomes extremely intractable so figuring out how to
19:16
actually do this is the fun part of the problem before we get into some of the methods I want to make a quick note
19:21
about reinforcement learning as I noted earlier I find reinforcement learning to be quite a challenging area there are
19:28
many different conflicting definitions about how these systems work and how they're actually trained I think many of
19:34
the details are actually specific choices that individual companies needed to make in their system design many of
19:40
these things often look quite different in open source than they might do within a kind of big reinforcement learning lab
19:47
with that in mind I think these choices are very important but for the sake of this talk I'm going to leave most of
19:53
them out there are particular choices about how to batch things how to do it on policy or off policy see how to use K
20:00
constraints to make sure your system doesn't go off the rails for most of the algorithms I'm going to talk about
20:05
though you can Implement them either in a simple way or in the more complex or scalable method it's not that these
20:12
details aren't important when I talk to experts they say these are some of the most important things to actually learn
20:18
and make these systems work it's just that I'm not the right person to tell you about them and I think they're interesting ideas here without going
20:24
into these systems in detail finally one last quote from open AI that I found sound really interesting they said that
20:30
when training a model for reasoning one thing that immediately jumps to mind is to have humans write out their thought
20:36
process and train on that when we saw that if you train the model using RL to generate and hone its own chain of
20:43
thoughts it can do even better than having humans right chain of thoughts for it that was the aha moment that you
20:50
could really scale this I think this quote is pretty amazing I imagine it was
20:55
quite shocking for the first time to see the model reason through a problem and get it correct I don't know maybe I'm a
21:01
sucker okay with that background we can actually get to the suspects for what might be going on here so I narrowed
21:07
this down to four different suspects guess and check process rewards search
21:13
or Alpha zero and learning to correct these aren't formally different areas
21:18
but when reading this literature I did find it a bit overwhelming there are many different papers that think about
21:24
these problems and well everyone's kind of convinced their method is what's really going on that being said I found
21:31
these four to be a useful outline for helping me think about this problem suspect one is the
21:38
simplest three steps we sample and chain of thoughts we check which ones were
21:43
successful and then we train on the good ones if we think about it in terms of
21:48
our picture we will independently sample some of the chains will go wildly off
21:54
and some will reach the solutions the solution ones are the ones we would like and so we can train them into our
22:01
language model if you're like me you might find it helpful to formalize what's going on here we can think about
22:07
this as a form of rejection sampling expectation maximization em is a very
22:13
traditional algorithm in machine learning and it's been applied to these sort of reinforcement learning algorithms for decades we can think of
22:20
the expectation step as running rejection sampling as we saw earlier in the talk and we can think of the
22:26
maximization step as fitting our l language models to the samples that fit with our posterior the more we run this
22:33
expectation step the closer to the true expectations we've calculated and the better our end-step will in getting to
22:40
the answer itself traditionally em is done in a batched offline process but
22:47
there are versions that are online or that can work with any other form of reinforcement learning given the
22:54
Simplicity of this method it's been discovered and noted to work many times
22:59
in MLP this works as a form of self-training this is a method that was
23:04
described in 1995 and later used to produce state-of-the-art syntactic parsers this
23:11
method of course has different names in different areas open AI refers to it as best of end training a popular recent
23:17
variant is called star or self-taught reasoning my formalization is from the
23:23
paper rest em in some sense the name here actually doesn't really matter too much I mostly list them all so you don't
23:29
get intimidated or think more is going on here than actually is and high level all these papers come to a similar
23:36
conclusion this method is simple but it works and it works pretty well you can
23:41
get relatively consistent improvements particularly in lower samples across many different problems if anything this
23:49
should be a required Baseline in most papers of course the Assumption here is that we have access to the verifier it
23:56
seems hard to actually scale the test time compute if we only have this during training of course we can try to use
24:03
what we've produced to also train the verifier since we have a lot of samples from rejection sampling we can further
24:11
try to train some sort of learned verifier that we can keep around at test time this idea which is often referred
24:17
to as amortization basically using a learned model to represent some sort of complex
24:23
system we can create our own sort of learn verifier at test time this could
24:28
then be used as part of Chain of Thought or for some sort of test time rejection sampling so is A1 just a guess and check
24:37
RL system well there's some signs it might be one thing that's neat is that
24:42
this approach is extremely simple and scalable one thing OpenAI has done in the past is just build larger more powerful
24:50
versions of things people thought worked reasonably well we also have seen positive results with this and
24:55
potentially with huge amounts of data collection with a verifier you could get a system like this to work really really
25:02
well what we're missing though is that there's no evidence that simply sampling will produce some of the chain of
25:08
thoughts that we saw earlier this seems like a big change to just have this happen automatically from our system in
25:16
addition the Assumption here is that if we do enough rejection sampling we'll get some good chains but for some of the
25:22
harder problems this seems really unlikely this seems computationally efficient or even impossible
25:28
so let's build a bit more structure into these systems this next section is on process rewards suspect two during chain
25:37
thought sampling we'll have some guidance that we'll both use to learn and to improve our trajectories we then
25:43
run the same process where we check if our final or partial versions are successful and train on the good ones
25:50
the term process rewards comes from two papers one from Google and one from open
25:55
AI in these papers they learn in early verification model which they call a PRM
26:00
or process reward model they show that learning this intermediate model can improve in rejection sampling compared
26:08
to a learned model that gets the full Solutions the graph on the right compares the Learned intermediate model
26:14
both to majority voting and to a model learned only on full Solutions note this
26:20
graph is not making any Claim about the learning process just that we're able to successfully complete more chain of
26:27
Thoughts by utilizing an intermediate learned verification function there are several ways for acquiring this process
26:34
reward model one might simply be to sample trajectories from your model and
26:40
utilize human annotators to label these another approach which is becoming more common in the literature is to take
26:47
partial chain of thoughts from your model and then perform rollouts these rollouts will tell us how good the Chain
26:53
of Thought is at a given time as we discussed earlier we can then and utilize these rollouts to train a
27:01
learned process reward model we'll call this RI this will give us a sense of how
27:06
good we are doing and can additionally be used at test time there are many ways we might choose to parameterise this
27:13
learned process re word model one interesting idea is to learn it as a
27:18
large language model the idea here is that the actual process reward model that's checking how well we're doing can
27:25
itself use chain of thought it might try to reason about individual steps and utilize them to decide upon the answer
27:33
what's important about this step is this is an idea that merges the generator and the verifier you can have a single model
27:40
that is both trying to do reasoning and also trying to verify this reasoning this is an idea that has begun to be an
27:46
Explorer recently in the literature where several approaches build these generative
27:51
verifiers on medium scale problems we can see that this learned process reward style seems to work in this paper known
27:59
as math sheeper they train a model utilizing rollouts they can show that in
28:05
this model they're both able to find better Solutions with their learned intermediate guide and they're also able
28:11
to learn a final model that's better at math this approach also brings the full
28:17
story into a bit more Focus if we're going to use a verifier that is also using Chain of Thought and we're going
28:24
to merge that into a single stream we can imagine alternating between generation and verification and using
28:31
that to improve our test time solution for example if we look back at one of the chain of thoughts I mentioned
28:37
earlier we see statements like is that a good explanation question mark while
28:42
traditionally we would think of this as part of the generator this might be part of a verifier that's been merged into
28:48
the same model it can move back and forth between generation and verification within a single language
28:55
mode so is this 01 well there's some evidence that these intermediate guides
29:00
are effective it also removes the challenge of just having a single learned verifier on the negative side we
29:07
haven't really seen anything yet that explains some of the advanced planning that we've seen this model do and we
29:13
don't really know how to fully do this combination of generator and process reward model into a single Chain of
29:20
Thought uh it's a compelling idea but there are a lot of details still remaining if I had to guess I would say
29:25
I personally think this is probably closest to what we might expect 01 to be it fits with the research papers that
29:32
open AI is publishing and some of the rumours about the Simplicity of the system itself that being said my
29:38
confidence is quite low and a lot of people I talked to think it's something quite a bit more advanced so let's look
29:44
at some more search based Solutions so in particular let's remind ourselves how alpha0 works this was a very important
29:52
paper in the history of deep learning and RL in this paper which was a follow-up to Alpha go they demonstrate
29:59
that a system completely taught with self-play could achieve expert level performance in a very hard task at a
30:07
casual level the way the system works is it plays games of go using a complex search algorithm then trains a neural
30:14
network based on the trajectories of this system it then uses that neural network to again play some more games
30:20
and iterates on this process this is the canonical example of success stories from
30:26
very simple rl-based algorithms and demonstrates scaling without the need for extensive expert demonstrations
30:33
there are several reasons this system is relevant to the discussion but one of the more recent ones is this work on
30:39
Alpha proof we don't have a lot of details behind how alpha proof works
30:44
just that it did extremely well at a very hard math competition and a blog post that says when presented with a
30:51
problem Alpha proof generates solution candidates and then proves or disproves them by searching over possible proof
30:58
steps in lean which is a proof assistant each proof that was found and verified
31:04
is used to reinforce Alpha proofs language model enhancing its ability to solve subsequent more challenging
31:10
problems so if you squint this does seem rather similar to some of the language
31:15
we saw in open ai's blog so how might this work well we're going to assume
31:21
there's going to be some self-play using some kind of guided search with exploration we'll then label the final
31:27
outcomes of these self-play games will train the guide and Generator and
31:33
iterate the terminology for this in the literature is known as expert iteration
31:38
it refers to this iterative process where an algorithm that combines a learn model plus a complex expert search the
31:45
goal is basically to distil the search process into the model we do this by
31:50
generating lots of samples utilizing our reward model to improve upon them and
31:56
then labeling the good ones we iterate on this process retraining both the generator as well as the guide model in
32:03
order to do better on the next iteration a popular way to do this with language modelling is to use a search algorithm
32:10
known as beam search and then to utilize a guide that will look very similar to our process reward model here our guide
32:17
will tell us how well we're doing by looking at our partial Chain of Thought So Far given this guide we can then perform
32:25
beam search the way beam search works is that at every step of the process we
32:30
expand out to all possible next chain of thoughts and then we just keep the top
32:36
four based on how well they're doing from our guide this will keep around four different possible solutions each
32:43
of the same length based on how close they are to producing a good final answer of course there's a lot of
32:49
details here I'm not specifying we have to determine how to expand to new possible chain of thoughts as well as
32:56
producing somewhat different chain of thoughts thoughts that might likely give us a different path we also have to
33:01
determine how to weigh the guide versus what the language model thinks is the next PATH but at high level you can see
33:08
how this expertise can then be trained into the model instead of a guide we might also consider using rollouts
33:15
directly in this example here we are running beam search at each step and
33:20
using rollouts at training time in order to tell us how well we're doing here's
33:25
our first step second step third step and here is our fourth step
33:32
in systems like AlphaGo these rollouts are combined with a learned guide function together these give us a good
33:38
sense of the problem and we learn how much we can actually trust our learned guide function versus explicit rollouts
33:46
remember our goal at test time is to remove the roll outs entirely several
33:51
recent papers have experimented with using these forms of expert iteration in order to produce good reasoning systems
33:59
again we're working on somewhat more mediums scale math problems but there is some preliminary evidence that there are
34:05
significant benefits from doing this form of search specifically we see large increases in accuracy compared with
34:12
doing the naive guess and check system here represented as star that we saw
34:17
earlier in the talk while beam search is a common approach for efficiently doing search with language models systems like
34:24
AlphaGo used much more complex forms of search for gambling in particular the
34:30
famous search algorithm used in those papers is known as MCTS Monte Carlo Tree
34:35
search this is a complex algorithm that combines search with exploration the way it works is that for
34:42
a given math problem we're going to start at the beginning we're then going to walk down our tree until we hit a
34:48
leaf node when we get to that node we'll expand five possible next steps so in
34:54
our case that node will consist of a partial Chain of Thought and the next five steps will be five other expansions
35:02
of steps we could try next we'll then pick one of those at random when we get
35:07
to that step we'll do a roll out of the next steps in the Chain of Thought
35:13
depending on whether these were successful or not we'll then update all of our parent notes based on the fact
35:19
that we ran that roll out and also how well it did we can continue this process
35:24
growing our tree and applying different rollouts the key to the algorithm is that we are
35:29
not simply just trying to reach the end as fast as possible but trying to explore different parts of the tree
35:37
let's look at a demo so in this example here we have run our selection process
35:42
and gone down the yellow nodes of the tree we then look at the bottom node
35:48
that Chain of Thought of three steps and we expand it to three next possible steps here we can see the expansion and
35:56
the yellow node represent present which of the expansions we pick to roll out we then run our roll outs here I'm showing
36:04
eight independent rollouts several of which reached the solution and several of which did not based on this roll out
36:11
we then update the node and all of its parents to tell them how well it did okay now we start the process again
36:20
we now need to select which node to expand next we've chosen a different set
36:25
of nodes to select and we've reached a different Leaf this selection process is
36:30
based both on which nodes won and which nodes were not yet explored we then
36:36
expand this node and we do another set of rollouts based on the success of these
36:42
rollouts we then update each of the parents as mentioned earlier the main benefit from this process is that when
36:49
we do selection we are basing that selection both on how well these nodes
36:54
did and also how well they were explored the benefit here is that we can explore
37:00
many different possible chain of thoughts and try to find ones that we may not have explored in the past so is
37:06
there some type of search algorithm being explored in 01 well it fits with the history of major demonstrated
37:12
results in RL and it's a particularly nice way of adding more training time
37:17
into the system remember we know that o1 is data efficient but they may have used
37:23
an incredible amount of compute in order to do training furthermore given that the chain of thoughts that we actually
37:29
are seeing from 01 look a little bit like search with properties like backtracking or highly outlining it's
37:36
plausible that those came into the model through something like training time search however there are some negatives
37:42
to this process it's much more complex algorithmically and it's costly to maintain open States compared to the
37:49
first two systems it does seem much harder to scale additionally we don't see anything about doing this sort of
37:56
complex training Tre search in any of the open AI release material the other thing that's interesting is that there
38:02
are a lot of papers kind of exploring MCTS for language modelling but at least in the open research Community we
38:09
haven't seen too many successes it seems like simpler methods work a bit better for these problems our final method is
38:16
learning to correct so to motivate these algorithms I want to note some of the differences between gam playing and
38:22
language in game playing there is a set of fixed moves and the main source of
38:28
exploration is to just explore these alternative moves in language there are really a lot of possibilities you can
38:35
get around some of this by sampling or fixing the next possible steps you might take but I think actually a lot of the
38:42
exploration should be in this process itself how you determine what are different next chain of thoughts which
38:48
ones will cause more exploration or cause more backtracking this motivates a series of methods on learning to correct
38:56
at a high level we're going to start start with some failed Chain of Thought that came from our system we're then
39:01
going to do some sort of search to find successful Corrections of this failure
39:07
based on this outcome we're then going to train on the entire process not just the correction but the original as well
39:14
a motivating example is work on self-correction the idea here is to isolate pairs of chain of thoughts we'll
39:21
call one Z Prime and the other Z double Prime these chain of thoughts are similar but one leads to a correct
39:28
answer and the other does not if we can isolate these two we can train a model
39:34
that can approve upon Z Prime in order to move more towards Z double Prime if
39:40
we can do this at scale we can hope to build this ability into the generator
39:45
itself this approach sounds simple but it actually has lots of challenges in practice one issue is that the model
39:52
will often just collapse if the First Chain of Thought was just not very good
39:57
it'll learn to ignore it and simply just directly try to generate the second one the other issue is that you have to be
40:03
extremely careful of distribution shift if you collect a lot of examples but they look different from what your
40:10
model's actually generating it might not actually self-correct in practice it'll
40:15
learn about your examples but not about what your model actually produces in mistakes one approach to get around this
40:22
issue is to try to be as UNP policy as possible in this setting we will
40:27
generate our Z Prime and then take all possible continuations from Z Prime to
40:33
the goal we will fix the model's original output and try to learn what
40:38
the correction would be from that given point there's some subtleties in getting this right for instance in the paper
40:45
listed they first run a training round where they only learn the correction part of the model then in a second stage
40:52
they also learn the original part of the model itself this is done to ensure sure that as the first part of the model
40:59
changes the correction part learns to adapt and continues producing good
41:04
Corrections when done correctly this approach beats both training on examples
41:10
as well as our guess and check approach it also scales better than simply pairing up the examples and learning to
41:16
correct from them of course our final goal is not individual corrections but Corrections applied repeatedly all in a
41:24
single chain this is the motivation behind tree of search in this work they
41:29
first find an optimal Zar they do this by applying a training time search
41:35
algorithm that produces the tree on the right this Tree finds a very good Chain
41:40
of Thought but also intermediate branches that go off in the wrong direction once we have this tree though
41:47
instead of training just on the path to the final answer we're going to instead linearize this tree into a single stream
41:55
the picture on the bottom shows schema automatically what's happening we're basically going to template construct a
42:01
synthetic Chain of Thought that goes in windy roads to get to the final answer
42:07
when we see actual backtracking in the tree this will become words in the Stream this gives us something that
42:14
looks like it's doing search even though it really is just a single path from the start to the end we can then train on
42:22
these streams in order to get search- like Behavior to summarize that a bit more informally we're going to try to
42:29
convert from a tree to a stream we do tree search to explore multiple paths
42:34
we'll convert this stream as a linear sequence and then we'll allow models to see their mistakes in the Stream this
42:42
sort of method might be combined with methods like learning to correct in order to make each individual Step
42:49
better and more on policy this approach is relatively complex but a lot of
42:55
experts I talked to were convinced that something like this is behind what 01 is doing so let's discuss the pros and cons
43:02
it seems like learning to correct and plan is a pretty important part of the process it's the first time we've seen
43:08
something that looks similar to what the actual chain of thoughts from 01 do it also seems plausible that this can be
43:15
used to induce search- like Behavior into a single test time model the
43:20
negative side this is also quite a complex training procedure there are many parts where the system could fall
43:25
apart or collapse this is because these Corrections require giving synthetic
43:30
examples or trying to keep the model on policy we also have limited empirical evidence so far in the open research
43:37
literature that this can induce this sort of interesting behavior in particular methods like stream of search
43:43
have only been applied on relatively simple problems still I think this is the most interesting of the potential
43:49
ideas and it would be really cool to see people build this in practice let me conclude by talking about some of the
43:55
implications of This research so first off the thing I care about most is actually replication as an open
44:03
source Community we need to get better at building some of these large scale rl-based systems and showing they can
44:10
really work it's critical to have open- Source versions of these models so that we can explore what's going on and to
44:17
build better or more efficient ones there are critical system aspects that differ for how open source systems might
44:23
look versus how the ones that companies are designed and and so it's quite possible the open source versions may
44:29
not actually look the same or maybe perform in different ways still even if we can't replicate exactly what open AI
44:37
did I think the fact that they demonstrated this result really should motivate the community to know that it's
44:42
possible and that we can build one of our own in addition there are incredibly exciting research implications behind
44:49
this work I want to talk about a couple areas that I think it's important to consider the first is that the last 5
44:55
years have really been dead dedicated to this idea of understanding what scaling means and how it changes how a language
45:02
model performs much of this has been somewhat mysterious these abilities come out of these models as we go I'm really
45:09
excited to understand how test time compute can be understood and how it changes some of these stories test time
45:16
compute seems much more transparent we can understand what the Chain of Thought is doing and to get a sense of how it
45:22
explains or contradicts the results it produces the other thing that's quite interesting is the fact that these
45:28
systems are bottlenecked by their inference time systems capabilities we've been focusing on inference to try
45:34
to make it cheaper or faster in a kind of chatbot setting but in this setting here inference really becomes
45:40
prioritized if you can make a system a thousand times faster that's say three or a magnitude of extra reasoning
45:45
ability that's a much more interesting capabilities change than simply kind of serving a model cheaper another thing
45:52
I'm really excited about is just to be done with prompting I think prompting has been intellectually really kind of
45:58
boring area and one that has kind of dominated the Practical use don't get me wrong I think prompting is really useful
46:05
and there's a lot of cool things you can do with it but there's not much more to say about it from a research perspective
46:10
I'm really interested in the move from prompting to some sort of formal specification if we can produce
46:16
interesting verifiers for hard problems and use language models to optimize against them that opens up all sorts of
46:22
interesting new areas of work next I think these these these models really open up many New Paths for evaluations
46:29
my group has been thinking a lot about evaluations that are just extremely hard and on tasks that we'd really like to do
46:36
but are way beyond the capability of even the best language models uh for instance we've been working on
46:41
benchmarks where you have to write entire new coding packages based just on their unit tests this sort of uh
46:47
superhuman evaluation becomes really exciting when you can have a model that just runs forever and takes that
46:53
feedback into account in terms of producing an answer I think we really need to think about evaluations in terms
46:58
of what we'd like these systems to do as opposed to just what we think the current generation can do and finally
47:03
maybe this one is obvious but throughout this entire talk I never really talked about neural networks at all the move to
47:10
search-based systems really is about how these systems are utilized what they
47:15
generate as their intermediate steps and how you might change or explore that this is very different than the sort of
47:21
interpretability that we've seen where you try to dive into the model itself and interpret its kind of Contin valued
47:27
weights I'd be really excited to see how this changes how we think about this problem how we understand our models or
47:33
what they do so thanks very much for listening um I have a GitHub page with
47:38
the full bibliography slides and issues from this talk I'll probably update it
47:44
for the next couple months as new papers come out or come in if you have any thoughts about where this is leading
47:50
research please leave them in the comments or an issue page or you can find me on Twitter thanks so much
