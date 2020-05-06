# Bayes and the worried parents
### A simple (morbid) example of how to use Bayes' theorem to update beliefs in light of new data. 

Notebook author: Patrick von Glehn

Example adapted from one given by a listener on the BBC 4 radio program More or Less. 


## The scenario

Every Sunday I go to my parents' house for dinner. After a lovely evening we say goodbye, I get in my car and travel 10 miles home. My parents always insist that I call them as soon as I get home and as I'm a good boy I almost always call, but occasionally, maybe twice a year, I forget. Last time this happened my parents phoned me after they couldn't bare waiting anymore and gave me an earful, saying that I had them worried sick that I'd had a terrible accident and died.

I tell them that the reverend Bayes says there was never any need to worry...

Bayesian statistics can be used to update our beliefs in light of new data. Let's see how my parents should have rationally updated their beliefs about my vital status given that I hadn't called after driving home from our Sunday dinner.

### Spoiler: with probability 99.99995% I was fine and just forgot to call


## Bayes' theorem

Using Bayes' theorem we can calculate, P(A|B): the probability of event A happening given event B has happened if we know (or can estimate) the following:


* P(A): the probability of event A happening under any circumstances (before seeing the data)
* P(B|A): the probability of event B happening given event A has happened
* P(B): the probability of event B happening under any circumstances
<blockquote>$P(A|B) = \frac{P(A)P(B|A)}{P(B)} $ </blockquote>


In our case we want to calculate P(A|B): the probability that I had a fatal car accident given that I didn't call my parents after leaving their house. To do the calculation we need:

* P(A): the probability of having a fatal car accident in a 10 mile stretch
* P(B|A): the probability that I don't call given that I have had a fatal car accident (certain)
* P(B): the probability that I don't call, either because I forgot or because I'm dead


Source for data: Department for Transport, Reported road casualties in Great Britain 2018 annual <a href="https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/834585/reported-road-casualties-annual-report-2018.pdf">report</a>



```python
# P(A): calculate the chance of dying in a 10 mile stretch from data about UK road fatalities
UK_fatalities_per_billion_miles_car = 1.8
P_dying_in_10_miles_car = UK_fatalities_per_billion_miles_car / 10.0**8 
P_not_dying_in_10_miles_car = 1 - P_dying_in_10_miles_car

# P(B|A): I definitely won't call if I'm dead
P_doesnt_call_if_dead = 1

# P(B): I forget to call about twice a year (52 weeks)
# The probability that I don't call under any circumstances =
# The probability that I arrive safely and then don't call + the probability that I die and then don't call

P_doesnt_call_if_not_dead = 2/52
P_doesnt_call = (P_not_dying_in_10_miles_car * P_doesnt_call_if_not_dead) + (P_dying_in_10_miles_car * P_doesnt_call_if_dead)

# Now let's calculate the final answer...
P_dead_if_doesnt_call = (P_dying_in_10_miles_car * P_doesnt_call_if_dead) / P_doesnt_call

chance_per_million = 1/(P_dead_if_doesnt_call * 10**6)

print(f"Probability I'm dead given that I haven't called = {P_dead_if_doesnt_call * 100 :.7f}% or one in {chance_per_million:.1f} million  ")
print(f"Probability I'm fine and just forgot to call = {(1 - P_dead_if_doesnt_call)*100}%")
```

    Probability I'm dead given that I haven't called = 0.0000468% or one in 2.1 million  
    Probability I'm fine and just forgot to call = 99.99995320002107%
    

# What about if I ride a motorbike instead of a car?


```python
UK_fatalities_per_billion_miles_motorbike = 119.7
P_dying_in_10_miles_motorbike = UK_fatalities_per_billion_miles_motorbike / 10.0**8 
P_not_dying_in_10_miles_motorbike = 1 - P_dying_in_10_miles_motorbike

P_doesnt_call_if_dead = 1
P_doesnt_call_if_not_dead = 2/52
P_doesnt_call = ((P_not_dying_in_10_miles_motorbike * P_doesnt_call_if_not_dead) 
                   + (P_dying_in_10_miles_motorbike * P_doesnt_call_if_dead))

P_dead_if_doesnt_call = (P_dying_in_10_miles_motorbike * P_doesnt_call_if_dead) / P_doesnt_call

chance_per_thousand = 1/(P_dead_if_doesnt_call * 10**3)

print(f"Probability I'm dead given that I haven't called = {P_dead_if_doesnt_call * 100 :.7f}% or one in {chance_per_thousand:.1f} thousand  ")
print(f"Probability I'm fine and just forgot to call = {(1 - P_dead_if_doesnt_call)*100}%")
```

    Probability I'm dead given that I haven't called = 0.0031121% or one in 32.1 thousand  
    Probability I'm fine and just forgot to call = 99.9968878931298%
    

# Conclusion:

If I don't call, don't worry I almost certainly just forgot. Although if I ride a motorbike regularly I am very significantly increasing my risk of having a fatal accident over the course of my life.
