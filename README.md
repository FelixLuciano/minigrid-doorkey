# Minigrid - Doorkey - Uma análise de modelos (DQN x PPO)

## Objetivos

O projeto tem como objetivo comparar o desempenho do ambiente single-agent [Minigrid - Doorkey](https://minigrid.farama.org/environments/minigrid/DoorKeyEnv/) utilizando os algoritmos DQN (Deep Q-Network) e PPO (Proximal Policy Optimization).

## Métodos

Para encontrar os resultados serão realizados treinamentos com diferentes números de episódios para cada algoritmo. Também será realizada a modificação do reward para que o agente consiga o reward não só chegando ao final mas realizando as ações necessárias para chegar lá, comparando os resultados. Para treinamento será utilizado a biblioteca stable baselines.

## Resultados esperados

Buscar entender a diferença entre os algoritmos para o treinamento do ambiente e como o reward em ações intermediárias interferem no aprendizado.

## Referências

- [Minigrid](https://minigrid.farama.org/environments/minigrid/DoorKeyEnv/)
- [Stable Baselines](https://docs.ray.io/en/latest/rllib/index.html](https://stable-baselines3.readthedocs.io/en/master/ )
