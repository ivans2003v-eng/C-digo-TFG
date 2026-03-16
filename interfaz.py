import sys
import json
from PyQt6.QtCore import pyqtSignal
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QComboBox, QFormLayout, QHBoxLayout, QSpacerItem, QSizePolicy, QFileDialog,
    QLineEdit, QPushButton, QLabel, QMessageBox, QScrollArea, QStackedWidget, QCheckBox, QSpinBox, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt
from qt_material import apply_stylesheet
import shutil


class MenuPage(QWidget):
    """Página con el menú de opciones."""
    change_page = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Menú Principal")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        # Crear botones para el menú
        self.new_database_button = QPushButton("Configuración de nueva base de datos")
        self.load_database_button = QPushButton("Modificación de la configuración de base de datos")
        self.execution_options_button = QPushButton("Modificación opciones de ejecución")

        # Conectar los botones con las funciones que manejan el cambio de pantalla
        self.new_database_button.clicked.connect(self.show_new_database_config)
        self.load_database_button.clicked.connect(self.show_load_database_config)
        self.execution_options_button.clicked.connect(self.show_execution_options)

        # Añadir botones al layout
        layout.addWidget(self.new_database_button)
        layout.addWidget(self.load_database_button)
        layout.addWidget(self.execution_options_button)

        self.setLayout(layout)

    def show_new_database_config(self):
        """Función para mostrar la configuración de nueva base de datos"""
        print("Ir a configurar nueva base de datos")
        self.change_page.emit(0)

    def show_load_database_config(self):
        """Función para mostrar la carga de base de datos"""
        print("Ir a modificar base de datos")
        self.change_page.emit(1)
        
    def show_execution_options(self):
        """Función para mostrar las opciones de ejecución"""
        print("Ir a opciones de ejecución")
        self.change_page.emit(2)


class MainWindow(QWidget):
    """Ventana principal que usa QStackedWidget para mostrar el menú y diferentes pantallas"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplicación")
        self.setGeometry(200, 200, 500, 300)
        self.setFixedSize(500, 300)

        # Crear el QStackedWidget para cambiar de pantallas
        self.stacked_widget = QStackedWidget(self)

        # Crear las páginas
        self.menu_page = MenuPage()  # Menú
        self.selection_page = SelectionBBDD()  # Página de selección de base de datos
        #self.modification_page = EEGModifyApp()  # Página de modificación de configuración de base de datos
        self.execution_options_page = ExecutionConfigPage()  # Página de modificación de configuración de base de datos
        self.menu_page.change_page.connect(self.change_page)

        # Añadir las páginas al QStackedWidget
        self.stacked_widget.addWidget(self.menu_page)
        self.stacked_widget.addWidget(self.selection_page)
        #self.stacked_widget.addWidget(self.modification_page)
        self.stacked_widget.addWidget(self.execution_options_page)

        # Establecer la página inicial
        self.stacked_widget.setCurrentWidget(self.menu_page)

        # Crear el layout principal
        layout = QVBoxLayout(self)
        layout.addWidget(self.stacked_widget)
        self.setLayout(layout)

    def change_page(self, index):
        """Cambia la página activa según el índice"""
        if index == 0:
            self.stacked_widget.setCurrentWidget(self.selection_page)
        elif index == 1:
            #self.stacked_widget.setCurrentWidget(self.modification_page)
            self.config_window = EEGModifyApp()
            self.config_window.show()
        elif index == 2:
            #self.stacked_widget.setCurrentWidget(self.execution_options_page)
            self.config_window = ExecutionConfigPage()
            self.config_window.show()


class SelectionBBDD(QWidget):
    """Ventana inicial para seleccionar la base de datos."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seleccionar Base de Datos")

        self.selected_data_dir = None
        self.selected_output_dir = None

        layout = QVBoxLayout()

        label = QLabel("Selecciona la base de datos:") 
        self.selector = QComboBox()
        self.selector.addItem("Seleccione una base de datos...")
        self.selector.addItems([
            "CHB-MIT EEG Database", 
            "SWEZ-ETHZ iEEG Database", 
            "Siena Scalp EEG Database", 
            "Otra base de datos"
        ])
        self.selector.currentIndexChanged.connect(self.on_selection_change)

        self.select_data_button = QPushButton("Seleccionar directorio de base de datos")
        self.select_data_button.clicked.connect(self.select_data_directory)
        self.select_data_button.hide()

        self.select_output_button = QPushButton("Seleccionar directorio de base de datos final")
        self.select_output_button.clicked.connect(self.select_output_directory)
        self.select_output_button.hide()

        self.confirm_button = QPushButton("Confirmar")
        self.confirm_button.clicked.connect(self.process_selection)

        layout.addWidget(label)
        layout.addWidget(self.selector)
        layout.addWidget(self.select_data_button)
        layout.addWidget(self.select_output_button)
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)

    def on_selection_change(self):
        """Muestra u oculta los botones de selección de directorios según la opción elegida."""
        selected = self.selector.currentText()
        if selected in ["CHB-MIT EEG Database", "SWEZ-ETHZ iEEG Database", "Siena Scalp EEG Database"]:
            self.select_data_button.show()
            self.select_output_button.show()
        else:
            self.select_data_button.hide()
            self.select_output_button.hide()

    def select_data_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar directorio de base de datos")
        if folder:
            self.selected_data_dir = folder

    def select_output_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar directorio de salida")
        if folder:
            self.selected_output_dir = folder

    def process_selection(self):
        selected = self.selector.currentText()

        if selected == "Otra base de datos":
            self.close()
            self.config_window = EEGConfigApp()
            self.config_window.show()
            return

        # Verifica que se hayan seleccionado ambos directorios
        if not self.selected_data_dir or not self.selected_output_dir and selected != "Otra base de datos":
            QMessageBox.warning(self, "Advertencia", "Debe seleccionar ambos directorios antes de continuar.")
            return

        # Define el nombre del archivo de configuración
        if selected == "CHB-MIT EEG Database":
            config_file = "config_chb_mit.json"
        elif selected == "SWEZ-ETHZ iEEG Database":
            config_file = "config_swec_thz.json"
        elif selected == "Siena Scalp EEG Database":
            config_file = "config_siena_scalp.json"
        elif selected == "Seleccione una base de datos...":
            QMessageBox.warning(self, "Advertencia", "Debe seleccionar una base de datos.")
            return
        else:
            config_file = None

        if config_file:
            source_path = os.path.join(os.getcwd(), config_file)
            target_path = os.path.join(os.getcwd(), "config.json")

            if os.path.exists(source_path):
                try:
                    shutil.copy(source_path, target_path)

                    # Ahora añade los directorios seleccionados al config.json
                    with open(target_path, "r") as f:
                        config = json.load(f)

                    config["bbdd_directory"] = self.selected_data_dir
                    config["result_directory"] = self.selected_output_dir

                    with open(target_path, "w") as f:
                        json.dump(config, f, indent=4)

                    QMessageBox.information(self, "Éxito", f"Configuración copiada y directorios guardados correctamente.")
                    QApplication.quit()

                except Exception as e:
                    QMessageBox.critical(self, "Error", f"No se pudo copiar el archivo de configuración: {str(e)}")
            else:
                QMessageBox.critical(self, "Error", f"No se encontró el archivo de configuración: {config_file}")


class EEGModifyApp(QWidget):
    def __init__(self, auto_load=True):
        super().__init__()
        self.setWindowTitle("Modificar Configuración EEG")
        self.setGeometry(50, 50, 820, 550)

        # Layout principal
        main_layout = QVBoxLayout(self)

        # Área de desplazamiento
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)
        content_widget = QWidget()
        self.layout = QVBoxLayout(content_widget)

        # Dropdown para seleccionar el formato
        self.format_label = QLabel("Selecciona el formato:")
        self.format_selector = QComboBox()
        self.format_selector.addItems(["EDF", "MAT"])
        self.format_selector.currentTextChanged.connect(self.update_form)

        # Formulario dinámico
        self.form_layout = QFormLayout()
        self.fields = {}

        # Botón para seleccionar el directorio de la base de datos
        self.select_folder_button = QPushButton("Seleccionar directorio de base de datos")
        self.select_folder_button.clicked.connect(self.select_directory_bbdd)
        self.layout.addWidget(self.select_folder_button)

        # Botón para modificar el directorio de la base de datos resultado
        self.select_folder_button = QPushButton("Seleccionar directorio de base de datos final")
        self.select_folder_button.clicked.connect(self.select_directory)
        self.layout.addWidget(self.select_folder_button)

        # Definir el nombre de la bbdd
        self.db_name_label = QLabel("Nombre de la base de datos:")
        self.db_name_input = QLineEdit()
        self.layout.addWidget(self.db_name_label)
        self.layout.addWidget(self.db_name_input)

        # Botón para guardar configuración
        self.save_button = QPushButton("Guardar Configuración")
        self.save_button.clicked.connect(self.save_config)

        # Configurar `QScrollArea`
        scroll_area.setWidget(content_widget)

        # Agregar widgets al layout
        self.layout.addWidget(self.format_label)
        self.layout.addWidget(self.format_selector)
        self.layout.addLayout(self.form_layout)
        self.layout.addWidget(self.save_button)

        self.setLayout(main_layout)

        # Cargar configuración si 'auto_load' es True
        if auto_load and os.path.exists("config.json"):
            self.load_config("config.json")
        else:
            self.update_form()  # Mostrar formulario vacío si no hay archivo JSON
    
    
    def select_directory_bbdd(self):
        """Muestra un cuadro de diálogo para seleccionar un directorio."""
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta", "")
        
        if folder:
            self.selected_directory_bbdd = folder
        else:
            QMessageBox.warning(self, "Error", "No se seleccionó ninguna carpeta.")

    def select_directory(self):
        """Muestra un cuadro de diálogo para seleccionar un directorio de base de datos."""
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de base de datos", "")
        
        if folder:
            self.selected_directory = folder
        else:
            QMessageBox.warning(self, "Error", "No se seleccionó ninguna carpeta.")

    def update_form(self):
        """Actualiza los campos del formulario según el formato seleccionado."""
        formato = self.format_selector.currentText()
        self.clear_form()

        if formato == "EDF":
            fields = {
                "fs": ("Frecuencia de muestreo", "Copie las palabras con las que indica la frecuencia de muestreo en el archivo de texto."),
                "channels": ("Nombres de los canales", "Copie las palabras comunes con las que nombra los canales en el archivo de texto."),
                "nmontages": ("Número de montajes", "Copie las palabras con las que indica el número de diferentes montajes de electrodos en el archivo de texto."),
                "file_name": ("Nombre del archivo", "Copie las palabras con las que indica el nombre del archivo en el archivo de texto."),
                "nseizures": ("Número de crisis", "Copie las palabras con las que indica el número de crisis epilépticas en el archivo de texto."),
                "start_seizure": ("Inicio de crisis", "Copie las palabras con las que indica el inicio de una crisis en el archivo de texto."),
                "end_seizure": ("Fin de crisis", "Copie las palabras con las que indica el fin de una crisis en el archivo de texto."),
                "time_format": ("Formato de tiempo", "Formato en el que están expresados los tiempos de inicio y fin de convulsión en el archivo de texto.", ["s", "hh.mm.ss", "hh:mm:ss"]),
                "tiempo_acumulado": ("Tiempo acumulado", "Indique si el inicio de la convulsión se tiene en cuenta desde el comienzo de cada archivo en el archivo de texto.", ["True", "False"]),
                "file_start": ("Inicio del archivo", "Copie las palabras con las que indica el inicio de un archivo en el archivo de texto."),
                "file_end": ("Fin del archivo", "Copie las palabras con las que indica el fin de un archivo en el archivo de texto.")
            }
        else:  # formato == "MAT"
            fields = {
                "fs": ("Frecuencia de muestreo", "Variable que contiene la frecuencia de muestreo."),
                "start_seizure": ("Inicio de crisis", "Variable que contiene los tiempos de inicio de las crisis."),
                "end_seizure": ("Fin de crisis", "Variable que contiene los tiempos de fin de las crisis."),
                "channels": ("Canales", "Variable que contiene la lista de canales utilizados."),
                "data": ("Datos EEG", "Variable que almacena los datos de las señales EEG.")
            }

        # Crear campos editables o combo box para `time_format` o `tiempo_acumulado`
        for key, (label_text, description, *options) in fields.items():
            # Agregar una etiqueta con la descripción del campo
            description_label = QLabel(f"<b>{label_text}:</b> {description}")
            self.form_layout.addRow(description_label)

            # Si el campo es `tiempo_acumulado`, usar un QComboBox con opciones True/False
            if key == "tiempo_acumulado":
                field = QComboBox()
                field.addItems(["True", "False"])  # Opciones True/False
            elif key == "time_format" and options:
                # Si el campo es `time_format`, usar un QComboBox con las opciones
                field = QComboBox()
                field.addItems(options[0])  # La lista de opciones está en `options[0]`
            else:
                field = QLineEdit()

            self.form_layout.addRow(key, field)
            self.fields[key] = field


    def clear_form(self):
        """Limpia el formulario antes de actualizarlo."""
        while self.form_layout.rowCount():
            self.form_layout.removeRow(0)
        self.fields.clear()

    def load_config(self, file_path):
        """Carga una configuración desde un archivo JSON."""
        try:
            with open(file_path, "r") as f:
                config = json.load(f)

            # Ajustar el formato según la configuración guardada
            formato = config.get("format", "EDF")
            self.format_selector.setCurrentText(formato)

            # Actualizar formulario con los valores del archivo
            self.update_form()
            for key, value in config.items():
                if key in self.fields:
                    field = self.fields[key]
                    if isinstance(field, QComboBox):
                        field.setCurrentText(str(value))
                    else:
                        field.setText(str(value))

            print("Configuración cargada correctamente:", config)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo cargar el archivo: {e}")
            print(f"Error al cargar el archivo: {e}")

    def save_config(self):
        """Guarda la configuración en un archivo JSON."""
        config = {"format": self.format_selector.currentText()}

        for key, field in self.fields.items():
            if isinstance(field, QComboBox):  # Si es un combo box, obtener el valor seleccionado
                value = field.currentText()
            else:
                value = field.text()

            config[key] = value if value else None  # Guarda None si está vacío

        # Para convertir el valor "True"/"False" en un valor booleano
        if config.get("tiempo_acumulado") == "True":
            config["tiempo_acumulado"] = True
        elif config.get("tiempo_acumulado") == "False":
            config["tiempo_acumulado"] = False
        
        config["result_directory"] = getattr(self, "selected_directory", None)
        config["bbdd_directory"] = getattr(self, "selected_directory_bbdd", None)
        
        
        # Obtener nombre de la base de datos
        db_name = self.db_name_input.text().strip()
        if not db_name:
            QMessageBox.warning(self, "Advertencia", "Debe ingresar un nombre para la base de datos.")
            return

        config["name"] = db_name  # Guardar el nombre de la base de datos

        for key in config.keys():
            if config[key] == "None":
                config[key] = None

        if config["result_directory"] == None:
            QMessageBox.warning(self, "Advertencia", "Debe seleccionar el directorio de base de datos final.")
            return

        if config["bbdd_directory"] == None:
            QMessageBox.warning(self, "Advertencia", "Debe seleccionar el directorio de base de datos original.")
            return
        
        try:
            with open("config.json", "w") as f:
                json.dump(config, f, indent=4)

            QMessageBox.information(self, "Éxito", "Configuración guardada correctamente.")
            print("Configuración guardada:", config)
            QApplication.quit() 

        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo guardar la configuración: {e}")
            print(f"Error al guardar el archivo: {e}")
            QApplication.quit() 


class EEGConfigApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Configuración de base de datos EEG")
        self.setGeometry(50, 50, 820, 550)

        # Layout principal
        main_layout = QVBoxLayout(self)

        # Área de desplazamiento
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  
        main_layout.addWidget(scroll_area)
        content_widget = QWidget()
        self.layout = QVBoxLayout(content_widget)

        # Dropdown para seleccionar el formato
        self.format_label = QLabel("Selecciona el formato:")
        self.format_selector = QComboBox()
        self.format_selector.addItems(["EDF", "MAT"])
        self.format_selector.currentTextChanged.connect(self.update_form)

        # Formulario dinámico
        self.form_layout = QFormLayout()
        self.fields = {}

        # Botón para seleccionar el directorio de la base de datos
        self.select_folder_button = QPushButton("Seleccionar directorio de base de datos")
        self.select_folder_button.clicked.connect(self.select_directory_bbdd)
        self.layout.addWidget(self.select_folder_button)

        # Botón para seleccionar el directorio de la base de datos resultado
        self.select_folder_button = QPushButton("Seleccionar directorio donde guardar la base de datos final")
        self.select_folder_button.clicked.connect(self.select_directory_result)
        self.layout.addWidget(self.select_folder_button)

        # Botón para guardar la configuración
        self.save_button = QPushButton("Guardar Configuración")
        self.save_button.clicked.connect(self.save_config)
        
        # Configurar `QScrollArea`
        scroll_area.setWidget(content_widget)

        # Agregar widgets al layout
        self.layout.addWidget(self.format_label)
        self.layout.addWidget(self.format_selector)
        self.layout.addLayout(self.form_layout)
        main_layout.addWidget(self.save_button)

        self.db_name_label = QLabel("Nombre de la base de datos:")
        self.db_name_input = QLineEdit()
        self.layout.addWidget(self.db_name_label)
        self.layout.addWidget(self.db_name_input)

        self.setLayout(self.layout)
        self.update_form()
    """
    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = QPixmap("C:/SONIA/TFG/eeg_channel_selection/logo.png")

        # Dibujar la imagen en la ventana y escalarla al tamaño de la ventana
        painter.drawPixmap(self.rect(), pixmap)
        """
    
    def select_directory_bbdd(self):
        """Muestra un cuadro de diálogo para seleccionar un directorio."""
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta", "")
        
        if folder:
            self.selected_directory_bbdd = folder
        else:
            QMessageBox.warning(self, "Error", "No se seleccionó ninguna carpeta.")

    def select_directory_result(self):
        """Muestra un cuadro de diálogo para seleccionar un directorio."""
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta", "")
        
        if folder:
            self.selected_directory_result = folder
        else:
            QMessageBox.warning(self, "Error", "No se seleccionó ninguna carpeta.")

    def update_form(self):
        """Actualiza los campos del formulario según el formato seleccionado."""
        formato = self.format_selector.currentText()
        self.clear_form()

        if formato == "EDF":
            fields = {
                "fs": ("Frecuencia de muestreo", "Copie las palabras con las que indica la frecuencia de muestreo en el archivo de texto."),
                "channels": ("Nombres de los canales", "Copie las palabras comunes con las que nombra los canales en el archivo de texto."),
                "nmontages": ("Número de montajes", "Copie las palabras con las que indica el número de diferentes montajes de electrodos en el archivo de texto."),
                "file_name": ("Nombre del archivo", "Copie las palabras con las que indica el nombre del archivo en el archivo de texto."),
                "nseizures": ("Número de crisis", "Copie las palabras con las que indica el número de crisis epilépticas en el archivo de texto."),
                "start_seizure": ("Inicio de crisis", "Copie las palabras con las que indica el inicio de una crisis en el archivo de texto."),
                "end_seizure": ("Fin de crisis", "Copie las palabras con las que indica el fin de una crisis en el archivo de texto."),
                "time_format": ("Formato de tiempo", "Formato en el que están expresados los tiempos de inicio y fin de convulsión en el archivo de texto.", ["s", "hh.mm.ss", "hh:mm:ss"]),
                "tiempo_acumulado": ("Tiempo acumulado", "Indique si el inicio de la convulsión se tiene en cuenta desde el comienzo de cada archivo en el archivo de texto.", ["True", "False"]),
                "file_start": ("Inicio del archivo", "Copie las palabras con las que indica el inicio de un archivo en el archivo de texto."),
                "file_end": ("Fin del archivo", "Copie las palabras con las que indica el fin de un archivo en el archivo de texto.")
            }
        else:  # formato == "MAT"
            fields = {
                "fs": ("Frecuencia de muestreo", "Variable que contiene la frecuencia de muestreo."),
                "start_seizure": ("Inicio de crisis", "Variable que contiene los tiempos de inicio de las crisis."),
                "end_seizure": ("Fin de crisis", "Variable que contiene los tiempos de fin de las crisis."),
                "channels": ("Canales", "Variable que contiene la lista de canales utilizados."),
                "data": ("Datos EEG", "Variable que almacena los datos de las señales EEG.")
            }

        # Crear campos editables o combo box para `time_format` o `tiempo_acumulado`
        for key, (label_text, description, *options) in fields.items():
            # Agregar una etiqueta con la descripción del campo
            description_label = QLabel(f"<b>{label_text}:</b> {description}")
            self.form_layout.addRow(description_label)

            # Si el campo es `tiempo_acumulado`, usar un QComboBox con opciones True/False
            if key == "tiempo_acumulado":
                field = QComboBox()
                field.addItems(["True", "False"])  # Opciones True/False
            elif key == "time_format" and options:
                # Si el campo es `time_format`, usar un QComboBox con las opciones
                field = QComboBox()
                field.addItems(options[0])  # La lista de opciones está en `options[0]`
            else:
                field = QLineEdit()

            self.form_layout.addRow(key, field)
            self.fields[key] = field


    def clear_form(self):
        """Limpia el formulario antes de actualizarlo."""
        while self.form_layout.rowCount():
            self.form_layout.removeRow(0)
        self.fields.clear()


    def save_config(self):
        """Guarda la configuración en un JSON y muestra un mensaje de confirmación."""
        formato = self.format_selector.currentText()
        config = {"format": formato}

        for key, field in self.fields.items():
            if isinstance(field, QComboBox):  # Si es un combo box, obtener el valor seleccionado
                value = field.currentText()
            else:
                value = field.text()

            config[key] = value if value else None  # Guarda None si está vacío

        # Para convertir el valor "True"/"False" en un valor booleano
        if config.get("tiempo_acumulado") == "True":
            config["tiempo_acumulado"] = True
        elif config.get("tiempo_acumulado") == "False":
            config["tiempo_acumulado"] = False
        
        config["result_directory"] = getattr(self, "selected_directory_result", None)

        if config["result_directory"] == None:
            QMessageBox.warning(self, "Advertencia", "Debe seleccionar la carpeta donde quiere que se guarde la base de datos generada.")
            return

        config["bbdd_directory"] = getattr(self, "selected_directory_bbdd", None)

        if config["bbdd_directory"] == None:
            QMessageBox.warning(self, "Advertencia", "Debe seleccionar la dirección de la base de datos.")
            return
        
        # Obtener nombre de la base de datos
        db_name = self.db_name_input.text().strip()
        if not db_name:
            QMessageBox.warning(self, "Advertencia", "Debe ingresar un nombre para la base de datos.")
            return

        config["name"] = db_name  # Guardar el nombre de la base de datos

        # Guardar en un archivo JSON
        try:
            with open("config.json", "w") as f:
                json.dump(config, f, indent=4)

            # Mostrar mensaje de confirmación
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setWindowTitle("Éxito")
            msg_box.setText("La configuración se ha guardado correctamente.")
            msg_box.exec()
            
            print("Configuración guardada correctamente:", config)
            QApplication.quit() 

        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo guardar la configuración: {str(e)}")
            print(f"Error al guardar el archivo: {e}")
            QApplication.quit() 


class ExecutionConfigPage(QWidget):
    """Página para configurar las opciones de ejecución del programa."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Configuración de Ejecución")
        self.setGeometry(20, 50, 700, 300)

        main_layout = QVBoxLayout()

        # Botón para seleccionar el directorio de la base de datos
        self.select_folder_button = QPushButton("Seleccionar directorio de base de datos")
        self.select_folder_button.clicked.connect(self.select_directory)
        main_layout.addWidget(self.select_folder_button)

        # Opción de trabajar con un solo paciente (deshabilitada al inicio)
        self.single_patient_checkbox = QCheckBox("Trabajar con un solo paciente")
        self.single_patient_checkbox.setEnabled(False)
        self.single_patient_checkbox.stateChanged.connect(self.toggle_patient_selection)
        main_layout.addWidget(self.single_patient_checkbox)

        # Selección de un solo paciente (deshabilitada al inicio)
        self.patient_selector = QComboBox()
        self.patient_selector.setEnabled(False)
        main_layout.addWidget(self.patient_selector)

        # Espaciador
        main_layout.addSpacing(10)

        # Opción de trabajar con todos los pacientes (deshabilitada al inicio)
        self.all_patients_checkbox = QCheckBox("Trabajar con todos los pacientes")
        self.all_patients_checkbox.setEnabled(False)
        self.all_patients_checkbox.stateChanged.connect(self.toggle_all_patients_selection)
        main_layout.addWidget(self.all_patients_checkbox)

        # Lista de pacientes (deshabilitada al inicio)
        self.patient_list = QListWidget()
        self.patient_list.setMinimumHeight(120)
        self.patient_list.setEnabled(False)
        self.patient_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        main_layout.addWidget(self.patient_list)

        # Espaciador
        main_layout.addSpacing(20)

        # Filtros
        main_layout.addWidget(QLabel("Filtros a aplicar:"))
        self.filters = {}
        filter_names = ["Filtro paso bajo", "Filtro paso alto", "Filtro notch"]
        for name in filter_names:
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)
            main_layout.addWidget(checkbox)
            self.filters[name] = checkbox

        # Espaciador
        main_layout.addSpacing(20)

        # Duración de la ventana en segundos
        main_layout.addWidget(QLabel("Duración de la ventana (segundos):"))
        self.window_duration = QSpinBox()
        self.window_duration.setRange(1, 60)
        self.window_duration.setValue(8)
        main_layout.addWidget(self.window_duration)

        # Espaciadores para centrar contenido
        main_layout.addStretch()

        """
        # Selección de características
        right_layout.addWidget(QLabel("Características a extraer:"))
        self.feature_list = QListWidget()
        self.feature_list.setMinimumHeight(500)

        feature_options = [
            'variance', 'std', 'zero_crossings', 'peak2peak', 'shannon_entropy',
            'variance-1', 'std-1', 'zero_crossings-1', 'peak2peak-1', 'shannon_entropy-1',
            'variance-2', 'std-2', 'zero_crossings-2', 'peak2peak-2', 'shannon_entropy-2',
            
            'total_energy', 'delta', 'theta', 'alpha', 'beta',
            'total_energy-1', 'delta-1', 'theta-1', 'alpha-1', 'beta-1',
            'total_energy-2', 'delta-2', 'theta-2', 'alpha-2', 'beta-2',
            
            'delta_theta', 'theta_alpha', 'alpha_beta', 'gammas',
            'delta_theta-1', 'theta_alpha-1', 'alpha_beta-1', 'gammas-1',
            'delta_theta-2', 'theta_alpha-2', 'alpha_beta-2', 'gammas-2'
        ]
        for feature in feature_options:
            item = QListWidgetItem(feature)
            item.setCheckState(Qt.CheckState.Checked)
            self.feature_list.addItem(item)
        right_layout.addWidget(self.feature_list)
        """
        # Botón para guardar configuración
        self.save_button = QPushButton("Guardar Configuración")
        self.save_button.clicked.connect(self.save_config)
        main_layout.addWidget(self.save_button)

        # Espaciador para centrar contenido
        main_layout.addStretch()

        # Añadir elementos al layout principal
        main_layout.addStretch()
        main_layout.addLayout(main_layout)
        main_layout.addStretch()
        self.setLayout(main_layout)

    def select_directory(self):
        """Muestra un cuadro de diálogo para seleccionar un directorio de base de datos."""
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de base de datos", "")
        
        if folder:
            self.selected_directory = folder
            self.load_patient_folders(folder)

            # Habilitar selección de pacientes
            self.single_patient_checkbox.setEnabled(True)
            self.all_patients_checkbox.setEnabled(True)
        else:
            QMessageBox.warning(self, "Error", "No se seleccionó ninguna carpeta.")

    def load_patient_folders(self, directory):
        """Carga los nombres de las carpetas de pacientes en el selector y la lista."""
        try:
            patient_folders = sorted([folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))])

            # Actualizamos las opciones
            self.patient_selector.clear()
            self.patient_list.clear()

            if patient_folders:
                self.patient_selector.addItems(patient_folders)

                # Solo llenamos la lista si la opción "Trabajar con todos los pacientes" está activada
                if self.all_patients_checkbox.isChecked():
                    for patient in patient_folders:
                        item = QListWidgetItem(patient)
                        item.setCheckState(Qt.CheckState.Checked)
                        self.patient_list.addItem(item)

        except Exception as e:
            print(f"Error al cargar las carpetas: {e}")
            QMessageBox.warning(self, "Error", "Hubo un problema al cargar las carpetas.")

    def toggle_patient_selection(self):
        """Habilita o deshabilita la selección de paciente según el checkbox."""
        if self.single_patient_checkbox.isChecked():
            self.all_patients_checkbox.setChecked(False)
            self.patient_selector.setEnabled(True)
            self.patient_list.setEnabled(False)
        else:
            self.patient_selector.setEnabled(False)

    def toggle_all_patients_selection(self):
        """Habilita o deshabilita la lista de pacientes según el checkbox."""
        if self.all_patients_checkbox.isChecked():
            self.single_patient_checkbox.setChecked(False)
            self.patient_selector.setEnabled(False)
            self.patient_list.setEnabled(True)

            # Llenar la lista con los pacientes disponibles si no está llena
            if self.patient_list.count() == 0:
                self.load_patient_folders(self.selected_directory)
        else:
            self.patient_list.setEnabled(False)

    def save_config(self):
        """Guarda la configuración seleccionada solo si hay al menos un paciente seleccionado."""
        selected_patients = []

        if self.single_patient_checkbox.isChecked():
            selected_patient = self.patient_selector.currentText()
            selected_patients = [selected_patient]
            if len(selected_patients) == 0:
                QMessageBox.warning(self, "Advertencia", "Debe seleccionar un paciente.")
                return

        elif self.all_patients_checkbox.isChecked():
            selected_patients = [
                self.patient_list.item(i).text() for i in range(self.patient_list.count())
                if self.patient_list.item(i).checkState() == Qt.CheckState.Checked
            ]
            if len(selected_patients) == 0:
                QMessageBox.warning(self, "Advertencia", "Debe seleccionar al menos un paciente en la lista.")
                return
        else:
            QMessageBox.warning(self, "Advertencia", "Debe seleccionar si desea trabajar con un solo paciente o con todos.")
            return

        config_execution = {
            "database_directory": getattr(self, "selected_directory", None),
            "selected_patients": selected_patients,
            "filters": {name: checkbox.isChecked() for name, checkbox in self.filters.items()},
            
            "window_duration": self.window_duration.value()
        }
        """
        "selected_features": [
            self.feature_list.item(i).text() for i in range(self.feature_list.count())
            if self.feature_list.item(i).checkState() == Qt.CheckState.Checked
        ],
        """
        try:
            with open("config_execution.json", "w") as f:
                json.dump(config_execution, f, indent=4)

            QMessageBox.information(self, "Éxito", "La configuración se ha guardado correctamente.")
            print("Configuración guardada correctamente:", config_execution)
            QApplication.quit()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo guardar la configuración: {str(e)}")
            print(f"Error al guardar el archivo: {e}")
            QApplication.quit()




# Ejecutar la aplicación
if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_pink.xml")
    selection_window = MainWindow()
    selection_window.show()

    sys.exit(app.exec())




"""
['dark_amber.xml', 'dark_blue.xml', 'dark_cyan.xml', 'dark_lightgreen.xml', 'dark_medical.xml', 
 'dark_pink.xml', 'dark_purple.xml', 'dark_red.xml', 'dark_teal.xml', 'dark_yellow.xml', 
 'light_amber.xml', 'light_blue.xml', 'light_blue_500.xml', 'light_cyan.xml', 'light_cyan_500.xml', 
 'light_lightgreen.xml', 'light_lightgreen_500.xml', 'light_orange.xml', 'light_pink.xml', 
 'light_pink_500.xml', 'light_purple.xml', 'light_purple_500.xml', 'light_red.xml', 
 'light_red_500.xml', 'light_teal.xml', 'light_teal_500.xml', 'light_yellow.xml']
"""
