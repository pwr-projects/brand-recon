# Rozpoznawanie znaków towarowych na obrazach statycznych

## Informacje

Program jest wynikiem projektu na analiza obrazów i wideo.

Jest napisany w języku Python. Należy go uruchamiać interpretatorem w wersji co najmnmiej 3.6.

## Instalacja

Instalacja niezbędnych do działania paczek:

### Opcjonalnie

Stworzyć wirtualne środowisko za pomocą na przykład **virtualenv**, aktywować.

```shell
$ source [virtualenv name]/bin/activate
```

Zainstalować paczki za pomocą komendy

```shell
$ pip install -r requirements.txt
```

lub 

```shell
# pip install -r requirements.txt
```

W przypadku Windowsa:

```
python -m pip install -r requirements.txt
```

## **UWAGA**

Do poprawnego działania programu należy mieć skompilowaną bibliotekę **OpenCV** wraz z płatnymi wersjami modułów dodatkowych (contrib modules), w których znajdują się zaimplementowane algorytmy SIFT oraz SURF.

Podczas powstawania projektu korzystano z OpenCV w wersji 4.0.0.
