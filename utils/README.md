This responsity is an embedding framework devoted to solving query answering in knowledge graph. And we believe the Wasserstein space could imporve the expressive power of entity and relation.<br>
Entity Embedding
---
We consider the probablity to explain the embedding, and endow the OT to measure the distance. The most promising embedding method is Wfr now(model3, model4).  The operator in logic query is modeled to t-norm, conjuntion \to t-min, disjunction \to t-max,   negation \to 1 - tensor.<br>
Projection Embedding
---
The key in this question is how to design relation projection. Because we did't use Neural Network to paticipate the logic operator, the embedding's effect is just rely on the relation projection. Thus we consider many types to model projection.<br>
In model3, we refer the most popular MLP. The input is relation and entity embedding. The Network has 3 depth, 1 hidden layer, could switch between batch_norm and layer_norm.
It has high score in 1p,2p,inp related to simple projection tasks.<br>
In model4, we refer GNN, model projection to matrix transformation for every relation.There two ways to prevent huge parameter, matrix decompostion and diagnoal matrix.<br>
W_r = \sum_{i}^{base_num} \alpha_{ri} W_i, the hypermeter is base_num.<br>
W_r = diag{W_{r1}, \cdots, diag_{rn}}, the size of W_{r1} is emb_grid.<br>
It has hign score in 3in,3inp,pi related to complex tasks.
Also, to improve the expressive ability, we try to mixture them and layer, but have poor effect.