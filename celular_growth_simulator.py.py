# celular_growth_simulator.py
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

class CellularGrowthSimulator:
    """
    Simulador interactivo del crecimiento de células cancerígenas con tratamiento
    """
    
    def __init__(self, K=800, N0_initial=50, r0=0.25, alpha0=0.01, Tf=40, Td=3.5):
        """
        Inicializa el simulador con parámetros por defecto
        
        Args:
            K: Capacidad de carga máxima
            N0_initial: Población inicial de células
            r0: Tasa de crecimiento inicial
            alpha0: Factor de efectividad del tratamiento
            Tf: Tiempo final de simulación (días)
            Td: Tiempo de duplicación (días)
        """
        self.K = K
        self.N0_initial = N0_initial
        self.r0 = r0
        self.alpha0 = alpha0
        self.Tf = Tf
        self.Td = Td
        
        # Inicializar widgets
        self._create_widgets()
        self._setup_ui()
        
    def _crecimiento_sin_tratamiento(self, t, N0, Td):
        """
        Modela el crecimiento celular sin tratamiento
        """
        return N0 * (2 ** (t/Td)) * np.exp(-0.02*t)
    
    def _crecimiento_con_tratamiento(self, t, N0, K, r, alpha):
        """
        Modela el crecimiento celular con tratamiento dirigido
        """
        A = (K - N0) / N0
        return K / (1 + A * np.exp(-r * t * (1 - alpha * t)))
    
    def _create_widgets(self):
        """Crea los widgets interactivos"""
        self.N0_slider = widgets.FloatSlider(
            value=self.N0_initial, min=10, max=200, step=10, 
            description='N₀:', style={'description_width': 'initial'}, 
            continuous_update=False
        )
        
        self.r_slider = widgets.FloatSlider(
            value=self.r0, min=0.05, max=1.0, step=0.05,
            description='r:', style={'description_width': 'initial'},
            continuous_update=False
        )
        
        self.alpha_slider = widgets.FloatSlider(
            value=self.alpha0, min=0.001, max=0.05, step=0.001,
            description='α:', style={'description_width': 'initial'},
            continuous_update=False
        )
        
        self.output_plot = widgets.Output()
    
    def _setup_ui(self):
        """Configura la interfaz de usuario"""
        self.controls = widgets.HBox(
            [self.N0_slider, self.r_slider, self.alpha_slider], 
            layout=widgets.Layout(flex_flow='row wrap', justify_content='space-around')
        )
        
        # Conectar eventos
        self.N0_slider.observe(self._update_plot, names='value')
        self.r_slider.observe(self._update_plot, names='value')
        self.alpha_slider.observe(self._update_plot, names='value')
    
    def _update_plot(self, change):
        """Actualiza el gráfico cuando cambian los parámetros"""
        self.output_plot.clear_output(wait=True)
        
        with self.output_plot:
            # Obtener valores actuales
            N0 = self.N0_slider.value
            r = self.r_slider.value
            alpha = self.alpha_slider.value
            
            # Crear gráfico
            t = np.linspace(0, self.Tf, 300)
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Calcular curvas
            crecimiento_sin = self._crecimiento_sin_tratamiento(t, N0, self.Td)
            crecimiento_con = self._crecimiento_con_tratamiento(t, N0, self.K, r, alpha)
            
            # Graficar
            ax.plot(t, crecimiento_sin, 'r-', linewidth=2, label='Sin tratamiento')
            ax.plot(t, crecimiento_con, 'b-', linewidth=2, label='Con tratamiento')
            ax.plot([0, self.Tf], [self.K/2, self.K/2], 'g--', linewidth=1.5, 
                    label=f'Mitad de capacidad máxima (K/2 = {self.K/2:.0f})')
            
            # Calcular A automáticamente
            A_auto = (self.K - N0) / N0
            
            # Calcular tiempo medio analítico
            t_half_analytic = self._calcular_tiempo_medio(N0, r, alpha, A_auto)
            
            if t_half_analytic is not None:
                ax.plot(t_half_analytic, self.K/2, 'go', markersize=8, 
                       label=f'T medio = {t_half_analytic:.2f} días')
                
                ax.text(0.02, 0.80, f'T analítico = {t_half_analytic:.2f} días', 
                       transform=ax.transAxes, fontsize=12, 
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Configurar gráfico
            ax.set_xlabel('Tiempo (días)')
            ax.set_ylabel('Número de células')
            ax.set_title('Crecimiento de Células Cancerígenas: Modelo Ajustado por N₀')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, self.K * 1.1)
            ax.set_xlim(0, self.Tf)
            
            # Mostrar parámetros
            param_text = f'Parámetros actuales:\nK = {self.K} (fijo)\nN₀ = {N0:.0f}, r = {r:.2f}, α = {alpha:.3f}\nA = {A_auto:.2f} (calculado)'
            ax.text(0.02, 0.98, param_text, transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
            
            plt.show()
    
    def _calcular_tiempo_medio(self, N0, r, alpha, A):
        """Calcula el tiempo cuando N_trat(t) = K/2"""
        try:
            a_param = -r * alpha
            b_param = r
            c_param = -np.log(A)
            
            discriminant = b_param**2 - 4*a_param*c_param
            if discriminant >= 0:
                t1 = (-b_param + np.sqrt(discriminant)) / (2*a_param)
                t2 = (-b_param - np.sqrt(discriminant)) / (2*a_param)
                
                t_half_analytic = max(t1, t2) if t1 > 0 and t2 > 0 else (t1 if t1 > 0 else t2)
                
                if t_half_analytic > 0 and t_half_analytic <= self.Tf:
                    return t_half_analytic
        except:
            pass
        
        return None
    
    def show_simulation(self):
        """
        Muestra la simulación interactiva
        
        Returns:
            None - muestra la interfaz interactiva
        """
        print("Simulación de Crecimiento Celular con Tratamiento")
        print("=" * 50)
        print(f"Parámetros fijos: K = {self.K}, Tiempo final = {self.Tf} días")
        print("A se calcula automáticamente como (K - N₀)/N₀")
        print()
        
        display(self.controls)
        display(self.output_plot)
        
        # Ejecutar primera actualización
        self._update_plot(None)

# Función de conveniencia para uso rápido
def mostrar_simulacion(K=800, N0=50, r=0.25, alpha=0.01, Tf=40):
    """
    Función rápida para mostrar la simulación con parámetros personalizados
    
    Args:
        K: Capacidad de carga máxima
        N0: Población inicial
        r: Tasa de crecimiento
        alpha: Factor de efectividad del tratamiento
        Tf: Tiempo final de simulación
    """
    simulator = CellularGrowthSimulator(K=K, N0_initial=N0, r0=r, alpha0=alpha, Tf=Tf)
    simulator.show_simulation()
    return simulator