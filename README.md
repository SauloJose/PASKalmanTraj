
# Filtro de Kalman - Simulador Interativo

Este projeto foi desenvolvido como parte da disciplina de **Processamento Adaptativo de Sinais** da **Universidade Federal de Campina Grande (UFCG)**, no período de 2026.1. O objetivo é simular, testar e visualizar o comportamento do Filtro de Kalman através de uma interface gráfica dinâmica.


## 🚀 Como Executar o Projeto

Para evitar conflitos de biblioteca, o ideal é rodar o projeto dentro de um ambiente virtual (`.venv`).

1. **Configure o ambiente virtual e ative-o:**
```bash
python -m venv .venv
# No Windows:
.venv\Scripts\activate
# No Linux/macOS:
source .venv/bin/activate
```

2. **Instale as dependências necessárias:**
```bash
pip install -r requirements.txt

```


3. **Rode a aplicação:**
Acesse a pasta raiz do executável e inicie o script principal:
```bash
cd FiltroKalman
python main.py

```



---

## 📂 Estrutura do Código

A arquitetura do projeto está dividida de forma modular para facilitar a manutenção:

* **`FiltroKalman/`**: Diretório principal do executável, onde fica o `main.py`.
* **`FiltroKalman/src/simulation/world.py`**: Onde a mágica acontece. Contém toda a lógica e a implementação matemática do Filtro de Kalman.
* **`FiltroKalman/src/gui/app.py`**: Código responsável pela interface gráfica (GUI) e pelo visualizador (*viewer*).
* **`FiltroKalman/src/generator/trajectories`**: Módulo que abriga o gerador de trajetórias utilizado nos testes.
* **`FiltroKalman/src/results/`**: Pasta onde o sistema salva os dados gerados pelas simulações.
* **`Artigos Base/`**: Repositório com os artigos científicos utilizados na fundamentação teórica deste projeto.
* **`Artigo do Projeto/`**: Tem o relatório com toda a implementação detalhada do código e das ideias, de forma sintetizada.
---

## 💡 Inspiração

Este projeto é uma evolução e melhoria de uma interface que desenvolvi anteriormente. Você pode conferir a versão original [clicando aqui](https://www.google.com/search?q=INSIRA_O_LINK_AQUI).

---

## 🔮 Futuras Implementações

O projeto ainda tem espaço para crescer. Algumas das melhorias planejadas são:

* [ ] Incorporar um **Filtro Adaptativo**.
* [ ] Refinar e polir o design da **Interface Gráfica**.
* [ ] Adicionar suporte a **outros tipos de filtros**.
* [ ] Trazer **mais interatividade** em tempo real para o usuário durante a simulação.

---

## ✉️ Contato

Se tiver qualquer dúvida sobre a implementação, lógica dos filtros ou quiser trocar uma ideia sobre o projeto, é só entrar em contato!

* **Desenvolvedor:** [GN0M10]
* **E-mail:** [saulo.jose.silva@ee.ufcg.edu.br]
* **GitHub:** [GN0M10](https://github.com/SauloJose)


