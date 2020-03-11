# Decision Trees

Idea of decision trees evolves from a simple fundamental of how humans make decisions. For example, below we have a decision tree around "If i wanted to go to lunch with my friend (John Snow) to a place that serves chiense food, the logic can be summarized in

![Simple Decision Tree](https://miro.medium.com/proxy/1*A79utV2rVyAzv5C3fQ0kmQ.png)

#### Notations: 
Box:  node
Top Box: Root
All notes at bottom layer: leaf nodes

Each node tests some attribute of the dataset and each branch corresponds to the value. 

## Picking the best attirbute: Information Gain & Entropy

#### Entropy: measure of disorder
randomness or impurity of a system / impurity of the cohort

![Entropy Formula](https://miro.medium.com/proxy/1*5ZB2z-FTkKZU2GtYZptMLg.png)

c: total number of attributes 
Pi: # of examples belonging to the ith class

Ex. We have two classes, red(R) and blue(B). For the first box, we have 25 red chocolates. The total number of chocolates is 50. So pi becomes 25 divided by 50. Same goes for blue class. Plug those values into entropy equation and we get this:

![Entropy Example](https://miro.medium.com/proxy/1*Rc4w1ojtIuPMdAZmvdDozg.png)

Calculating it out gives us entropy of 1 meaning both cateogries are equally distributed

#### Information Gain: based on the decrease in entropy after dataset is split on attribute

Refer to example below to understand how information gain helps pick the best attribute


### Building the Decision Tree
First, let’s take our chocolate example and add a few extra details. We already know that the box 1 has 25 red chocolates and 25 blue ones. Now, we will also consider the brand of chocolates. Among red ones, 15 are Snickers and 10 are Kit Kats. In blue ones, 20 are Kit Kats and 5 are Snickers. Let’s assume we only want to eat red Snickers. Here, red Snickers (15) become positive examples and everything else like blue Snickers and red Kit Kats are negative examples.

Now, the entropy of the dataset with respect to our classes (eat/not eat) is:
![Entropy Example](https://miro.medium.com/proxy/1*GTfn9dwc_5KIyXxqFQYGxg.png)

Let’s take a look back now — we have 50 chocolates. If we look at the attribute color, we have 25 red and 25 blue ones. If we look at the attribute brand, we have 20 Snickers and 30 Kit Kats.

To build the tree, we need to pick one of these attributes for the root node. And we want to pick the one with the highest information gain. Let’s calculate information gain for attributes to see the algorithm in action.

Information gain with respect to color would be:
![Information Gain](https://miro.medium.com/proxy/1*fGB_u9WjYY7bEqwsxynCQg.png)

We just calculated the entropy of chocolates with respect to class, which is 0.8812. For entropy of red chocolates, we want to eat 15 Snickers but not 10 Kit Kats. The entropy for red chocolates is:
![Entropy of Red chocolates](https://miro.medium.com/proxy/1*ovrVjLPnhMy3KgzeeIpt5g.png)

For blue chocolates, we don’t want to eat them at all. So entropy is 0.

Our information gain calculation now becomes:
![Information Gain](https://miro.medium.com/proxy/1*JLPIu1JsVk81o3sItLlKbA.png)

If we split on color, information gain is 0.3958.

Let’s look at the brand now. We want to eat 15 out of 20 Snickers. We don’t want to eat any Kit Kats. The entropy for Snickers is:

![Entropy for Snickers](https://miro.medium.com/proxy/1*tRRVACuD03m4mIpPYyUVuQ.png)


We don’t want to eat Kit Kats at all, so Entropy is 0. Information gain:
![Information Gain](https://miro.medium.com/proxy/1*PIKu8876zb1wee1k3dezWw.png)


Information gain for the split on brand is 0.5567.

Since information gain for brand is larger, we will split based on brand. For the next level, we only have color left. We can easily split based on color without having to do any calculations. Our decision tree will look like this:
![Decision Tree](https://miro.medium.com/proxy/1*YE2C-_w5VpdvwefcgjVVmA.png)

## Evaluating your tree
The most decisive factor for the efficiency of a decision tree is the efficiency of its splitting process. We split at each node in such a way that the resulting purity is maximum. Well, purity just refers to how well we can segregate the classes and increase our knowledge by the split performed. An image is worth a thousand words. Have a look at the image below for some intuition:

![Gini Coeff](https://dimensionless.in/wp-content/uploads/RandomForest_blog_files/figure-html/gini.png)




## Resources
MIT Deck: https://www.cs.cmu.edu/afs/cs/academic/class/15381-s07/www/slides/041007decisionTrees1.pdf


