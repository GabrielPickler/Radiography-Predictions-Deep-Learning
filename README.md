# Predições de Radiografias COVID-19 utilizando Deep Learning

# Configurando o Ambiente

Requerimentos:
- Python 3.8
- Pycharm
- As dependências estão no arquivo requirements.txt, para instalar as versões basta clonar o projeto que a própria IDE já baixa todas as dependências. (Qualquer dúvida <a href="https://www.jetbrains.com/help/pycharm/managing-dependencies.html#apply_dependencies">🔗 clique aqui</a>)
- TensorFlow (Caso tenha problemas para rodar o projeto, verifique os requisitos do tensorflow <a href="https://www.tensorflow.org/install/pip?hl=pt-br">aqui</a>)

# Configurando o dataset

O dataset utilizado no projeto foi retirado desse site: https://www.kaggle.com/tawsifurrahman/covid19-radiography-database. 
Para utiliza-lo você deve baixar o pacote de imagens, e no projeto você irá criar uma pasta chamada datasets e dentro dessa pasta você irá colocar as três pastas contendo todas as imagens radiográficas: COVID, NORMAL, VIRAL PNEUMONIA.
Então irá ficar assim o caminho das pastas:
```
datasets
   --> COVID
        --> COVID (1).png
        --> COVID (2).png
        ...
   --> NORMAL
        --> NORMAL (1).png
        --> NORMAL (2).png
        ...
   --> VIRAL PNEUMONIA
        --> VIRAL PNEUMONIA (1).png
        --> VIRAL PNEUMONIA (2).png
        ...
```
        
Em seguida, seu projeto estará pronto para ser utilizado. Qualquer dúvidas entre em contato comigo no seguinte email: gabriel.pickler@unisul.br
