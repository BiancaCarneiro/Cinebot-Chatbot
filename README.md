# Chatbot-NLP
Nesse projeto vamos criar um chatbot que funcionaria como um assistente virutal que serve para dar informações e vender ingressos.

Nosso chatbot conta com detecção de intenções e utiliza um web-scraper para coletar informações do cinema do shopping 'Rio Mar Recife'. A partir dessas informações ele possui as funções de mostrar os filmes em cartaz e de mostrar os preços dos ingressos. Além disso ele é capaz de entender o filme que você quer assistir e o horário que deseja assistir ele para, enfim, poder reservar.

Além do chatbot, em outro segmento do projeto, utilizamos diferentes arquiteturas (LSTM, Transformers, CNN e SVM) para conseguir detectar intenções dos textos presentes em nosso dataset. 
Fora isso, também construimos um extrator de entidades para que seja possível extrair entidades ,como nome de filmes, genêro, etc, que são importantes para esse contexto. O modelo utilizado para tal detecção foi uma LSTM.

O nosso dataset foi feito com o Chat-GPT para criar frases que faziam sentido no nosso contexto de um chatbot de ajuda e compra de filmes de um cinema.
