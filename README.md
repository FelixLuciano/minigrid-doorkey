# FLATLANDS - Uma análise de modelos (DQN x PPO)

## Objetivos

O projeto tem como objetivo comparar o desempenho do ambiente multi-agent [Flatlands](https://flatland.aicrowd.com/intro.html) utilizando os algoritmos DQN (Deep Q-Network) e PPO (Proximal Policy Optimization) com diferentes números de agentes.

## Métodos

Para encontrar os resultados serão realizados treinamentos com diferentes números de trens(agentes) para cada algoritmo e a partir da curva de aprendizado resultante será comparado o desempenho dos algoritmos em diferentes cenários. Também será comparado o agente pós-treinamento verificar qual o tempo gasto para o agente chegar no estado final, e para isso serão executados vários testes e comparados os resultados, verificando min, max e média. Para o treinamento utilizaremos a biblioteca RLib.

## Resultados esperados

Buscar entender qual a influência do número de agentes no tempo de convergência de cada algoritmo. E entender como cada algoritmo influencia na jornada final de cada algoritmo.

## Referências

- [Flatlands](https://flatland.aicrowd.com/intro.html)
- [RLib](https://docs.ray.io/en/latest/rllib/index.html)
