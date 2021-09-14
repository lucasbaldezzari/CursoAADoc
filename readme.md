# Curso *Aprendizaje Automático y Aplicaciones* - Doctorado en Ingeniería

- [x] Repositorio para almacenar contenido de resolución de actividades del curso.

### Instalar un Enviroment en Conda para trabajar durante el taller

Pasos propuestos a seguir:

- _Abrir la consola de Anaconda, de Windows o Linux._
- _Ejecutar:_ conda install --name base nb_conda_kernels
- _Moverse hasta el directorio donde se almacenará el trabajo_
- _Ejecutar:_ conda env update --file dependencias.yml (Nota: el nombre por defecto del enviroment es "CursoAADoc", para cambiarlo debe editarse el archivo dependencias.yml)
- _Activar el ambiente:_ conda activate CursoAADoc

Al finalizar el proceso deberían ver un mensaje similar a este:

_To activate this environment, use_

     $ conda activate CursoAADoc

_To deactivate an active environment, use_

     $ conda deactivate

### Seteando Jupyter

Agregando mi enviroment a Jupyter notebook
- [x] conda install -c anaconda ipykernel

Seteando directorio de trabajo
- [x] jupyter notebook --notebook-dir E:\CursoAADoc

- [Info](https://towardsdatascience.com/get-your-conda-environment-to-show-in-jupyter-notebooks-the-easy-way-17010b76e874)

### Guía Markdown

- [The Ultimate Markdown Guide](https://medium.com/analytics-vidhya/the-ultimate-markdown-guide-for-jupyter-notebook-d5e5abf728fd)

### Agregando kernel a Jupyter

http://echrislynch.com/2019/02/01/adding-an-environment-to-jupyter-notebooks/
https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084