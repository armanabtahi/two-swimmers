import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrow
from matplotlib.widgets import Button
import numpy as np
import threading
import joblib
# load json and create model

loaded_model =  joblib.load("web_application/swimmer_model.joblib")

# position = [x_a y_a theta_a x_b y_b theta_b]
# X = [r phi theta]
# Y = [U_a_par U_a_norm W_a_per U_b_par U_b_norm W_b_per]

def input(position,types):
    x_a, y_a, theta_a, x_b, y_b, theta_b=position
    alpha_sign,beta_sign=types
    
    r=np.sqrt((x_a-x_b)**2+(y_a-y_b)**2)
    theta=theta_b-theta_a+np.pi/2
    
    if x_b==x_a:
        theta_r=np.pi/2
    else:
        theta_r=np.arctan((y_b-y_a)/(x_b-x_a))
    
    phi=np.pi/2+theta_r-theta_a
    
    #correction
    if r<=2.0001:
        diff=2.0001-r
        x_a=x_a-diff/2*np.cos(theta_r)
        y_a=y_a-diff/2*np.sin(theta_r)
        x_b=x_b+diff/2*np.cos(theta_r)
        y_b=y_b+diff/2*np.sin(theta_r)
        r=2.0001
        position=[x_a, y_a, theta_a, x_b, y_b, theta_b]
        
    X=np.array([[alpha_sign,beta_sign,r,phi,theta]])
    
    return X,position

def geometry_update(position,Y,dt):
    x_a, y_a, theta_a, x_b, y_b, theta_b = position
    U_a_par, U_a_norm, W_a_per, U_b_par, U_b_norm, W_b_per = Y.flatten().tolist()
    
    r=np.sqrt((x_a-x_b)**2+(y_a-y_b)**2)
    parallel=[x_b-x_a , y_b-y_a]/r
    normal=[y_a-y_b , x_b-x_a]/r
    
    U_a=U_a_par*parallel+U_a_norm*normal
    U_b=U_b_par*parallel+U_b_norm*normal
    
    
    x_a=x_a+U_a[0]*dt
    y_a=y_a+U_a[1]*dt
    x_b=x_b+U_b[0]*dt
    y_b=y_b+U_b[1]*dt
    theta_a=theta_a+W_a_per*dt
    theta_b=theta_b+W_b_per*dt
    
    position = [x_a, y_a, theta_a, x_b, y_b, theta_b ]
    
    return position

def update_figure(position,i,types):
    global ax,fig,objects
    if types[0]==1:
        alpha_color='r'
    elif types[0]==-1:
        alpha_color='#00ffff'
    else :
         alpha_color='black'
        
    if types[1]==1:
        beta_color='r'
    elif types[1]==-1:
        beta_color='#00ffff'
    else :
         beta_color='black'

    alpha_sign,beta_sign=types
    x_a, y_a, theta_a, x_b, y_b, theta_b = position
    objects["circles"][0].set_center((x_a, y_a))
    objects["circles"][1].set_center((x_b, y_b))
    objects["centers"][0].set_center((x_a, y_a))
    objects["centers"][1].set_center((x_b, y_b))
    objects["arcs"][0].set_center((x_a, y_a))
    objects["arcs"][1].set_center((x_b, y_b))
    objects["arcs"][0].set_angle(-90+theta_a*180/np.pi)
    objects["arcs"][1].set_angle(-90+theta_b*180/np.pi)
    traj_a=ax.add_patch(plt.Circle((x_a, y_a), 0.02, color=alpha_color, alpha=0.5))
    traj_b=ax.add_patch(plt.Circle((x_b, y_b), 0.02, color=beta_color, alpha=0.5))

    objects["traj"].append(traj_a)
    objects["traj"].append(traj_b)


    fig.canvas.draw()
    #time.sleep(0.0001)

def add_objects(int_pos):
    global ax, fig,objects
    x_a, y_a,theta_a,x_b, y_b,theta_b=int_pos
    circle_a=ax.add_patch(plt.Circle((x_a, y_a), 1, color="black", alpha=0.5))
    circle_b=ax.add_patch(plt.Circle((x_b, y_b), 1, color="black", alpha=0.5))
    arc_a=ax.add_patch(Arc((x_a, y_a), 2, 2,theta1=theta_a*180/np.pi-25, theta2=theta_a*180/np.pi+25, edgecolor='black', lw=3))
    arc_b=ax.add_patch(Arc((x_b, y_b), 2, 2,theta1=theta_b*180/np.pi-25, theta2=theta_b*180/np.pi+25, edgecolor='black', lw=3))
    center_a=ax.add_patch(plt.Circle((x_a, y_a), 0.03, color='black'))
    center_b=ax.add_patch(plt.Circle((x_b, y_b), 0.03, color='black'))
    dot_assist_a=ax.add_patch(plt.Circle((x_a, y_a+1.5), 0.15, color='black',alpha=0.6))
    dot_assist_b=ax.add_patch(plt.Circle((x_b, y_b+1.5), 0.15, color='black',alpha=0.6))
    line_assist_a=plt.arrow(x_a, y_a,0,1.5, ls="--",color='black' ,alpha=0.6)
    line_assist_b=plt.arrow(x_b, y_b,0,1.5, ls="--",color='black' ,alpha=0.6)
    traj_a=ax.add_patch(plt.Circle((x_a, y_a), 0.02, color='black', alpha=0.5))
    traj_b=ax.add_patch(plt.Circle((x_b, y_b), 0.02, color='black', alpha=0.5))
    plt.draw()
    circles = [circle_a, circle_b]
    arcs = [arc_a, arc_b]
    centers = [center_a, center_b]
    dots_assist = [dot_assist_a, dot_assist_b]
    lines_assist = [line_assist_a, line_assist_b]
    trajs =[traj_a,traj_b]
    objects={}
    objects["circles"]=circles
    objects["arcs"]=arcs
    objects["centers"]=centers
    objects["dots_assist"]=dots_assist
    objects["lines_assist"]=lines_assist
    objects["traj"]=trajs
    




def initiate_swimmers(figsize=(10, 10),xlim=(-5,5),ylim=(-5,5),circ_1_i=(-1.5,0),circ_2_i=(1.5,0)):
    global ax, fig,objects
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_aspect('equal')
    
    x_a, y_a=circ_1_i
    x_b, y_b=circ_2_i
    theta_a=np.pi/2
    theta_b=np.pi/2
    
    int_pos=[x_a, y_a,theta_a,x_b, y_b,theta_b]
    
    add_objects(int_pos)
    

def initial_position():
    global objects
    (xc_a,yc_a)=objects["circles"][0].get_center()
    (xc_b,yc_b)=objects["circles"][1].get_center()

    (x_a,y_a)=objects["dots_assist"][0].get_center()
    (x_b,y_b)=objects["dots_assist"][1].get_center()
    v1 = [1, 0]
    v2_a = [x_a-xc_a,  y_a-yc_a]
    v2_a = v2_a/np.linalg.norm(v2_a)
    v2_b = [x_b-xc_b,  y_b-yc_b]
    v2_b = v2_b/np.linalg.norm(v2_b)
    if v2_a[1]<0:
        theta_a = 2*np.pi-np.arccos(np.dot(v1,v2_a))
    else: 
        theta_a = np.arccos(np.dot(v1,v2_a))
    if v2_b[1]<0:
        theta_b = 2*np.pi-np.arccos(np.dot(v1,v2_b))
    else: 
        theta_b = np.arccos(np.dot(v1,v2_b))    

    int_pos=[xc_a,yc_a,theta_a,xc_b,yc_b,theta_b]
    



    return int_pos

def start_interface():
    global types,ax
    types=[0,0]
    initiate_swimmers(ylim=(-2,8))
    ax.set_title(" Please locate swimmers, their orientation and specify their types.")
    
    
    bottom_buttons()
    drag_objects()
    upper_buttons()   
    
    global thread 
    thread = threading.Thread(target=start_trajectories, args=())

    plt.show()
    

def start_trajectories(dt=0.1):
    global ax, fig,objects,dcs,bs
    for i,dc in dcs.items():
        dc.disconnect()

    for i,b in bs.items():
        b.disconnect_events()

    objects["dots_assist"][0].remove()
    objects["dots_assist"][1].remove()
    objects["lines_assist"][0].remove()
    objects["lines_assist"][1].remove()

    int_pos=initial_position()
    deploy(int_pos,types,dt)
    
def deploy(initial_position,types,dt):
    global ax, fig,objects,dcs,bs
    position={}
    position[0]=initial_position
    time_passed=0
    i=0
    global RESET
    while not RESET:
        global pause
        global restart


        if not pause:
            X,position[i]=input(position[i],types)
            Y=loaded_model.predict(X)
            update_figure( position[i],i,types)
            position[i+1]=geometry_update(position[i],Y,dt)
            time_passed+=dt
            ax.set_title("Time passed "+f"{time_passed:.1f}")
            i+=1
        
            if restart:
                restart=False
                i=0
                time_passed=0
                ax.set_title("Time passed "+f"{time_passed:.1f}")
                for item in objects["traj"][2:]:
                        #objects["traj"].pop(index)
                        item.remove()
                        objects["traj"].remove(item)


def bottom_buttons():
    global bs,objects
    callback = Index(objects["circles"])
    axes={}
    axes["a_push"] = plt.axes([0.1, 0.04, 0.11, 0.04])
    axes["a_pull"] = plt.axes([0.21, 0.04, 0.11, 0.04])
    axes["a_neut"] = plt.axes([0.32, 0.04, 0.11, 0.04])
    axes["b_push"] = plt.axes([0.6, 0.04, 0.11, 0.04])
    axes["b_pull"] = plt.axes([0.71, 0.04, 0.11, 0.04])
    axes["b_neut"] = plt.axes([0.82, 0.04, 0.11, 0.04])
    
    bs={}
    bs["a_push"] = Button(axes["a_push"], 'alpha: pusher',hovercolor="#00ffff")
    bs["a_push"].on_clicked(callback.alpha_type_pusher)
    bs["a_pull"] = Button(axes["a_pull"], 'alpha: puller',hovercolor="#F08080")
    bs["a_pull"].on_clicked(callback.alpha_type_puller)
    bs["a_neut"] = Button(axes["a_neut"], 'alpha: neutral')
    bs["a_neut"].on_clicked(callback.alpha_type_neutral)
    bs["b_push"] = Button(axes["b_push"], 'beta: pusher',hovercolor="#00ffff")
    bs["b_push"].on_clicked(callback.beta_type_pusher)
    bs["b_pull"] = Button(axes["b_pull"], 'beta: puller',hovercolor="#F08080")
    bs["b_pull"].on_clicked(callback.beta_type_puller)
    bs["b_neut"] = Button(axes["b_neut"], 'beta: neutral')
    bs["b_neut"].on_clicked(callback.beta_type_neutral)
    
def drag_objects():
    global objects,dcs
    dcs={}
    dcs[0]=DraggableCircle(objects["centers"][0],objects["dots_assist"][0],objects["circles"][0],objects["arcs"][0],objects["lines_assist"][0])
    dcs[1]=DraggableCircle(objects["centers"][1],objects["dots_assist"][1],objects["circles"][1],objects["arcs"][1],objects["lines_assist"][1])
    dcs[2]=RotateCircle(objects["dots_assist"][0],objects["circles"][0],objects["arcs"][0],objects["lines_assist"][0])
    dcs[3]=RotateCircle(objects["dots_assist"][1],objects["circles"][1],objects["arcs"][1],objects["lines_assist"][1])
    
    for i,dc in dcs.items():
        dc.connect()

def upper_buttons():
    global bs2
    callback_start = Index_start()  
    axes={}
    axes["start"] = plt.axes([0.15, 0.8, 0.13, 0.04])
    axes["pause"] = plt.axes([0.15, 0.75, 0.13, 0.04])
    axes["reset"] = plt.axes([0.15, 0.7, 0.13, 0.04])
    
    bs2={}
    bs2["start"] =Button(axes["start"], 'START / RESTART',hovercolor="r")
    bs2["start"].on_clicked(callback_start.start_button)
    
    bs2["pause"]=Button(axes["pause"], 'PAUSE / RESUME',hovercolor="r")
    bs2["pause"].on_clicked(callback_start.pause_button)
    
    bs2["reset"]=Button(axes["reset"], 'RESET',hovercolor="r")
    bs2["reset"].on_clicked(callback_start.reset_button)
    
    
    
    
class DraggableCircle:
    def __init__(self, cent,dot,circ,arc,line):
        self.circ = circ
        self.press = None
        self.dot = dot
        self.arc = arc
        self.line = line
        self.cent = cent
        

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.circ.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.circ.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.circ.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        """Check whether mouse is over us; if so, store some data."""
        if event.inaxes != self.circ.axes:
            return
        contains, attrd = self.circ.contains(event)
        if not contains:
            return
        print('event contains', self.circ.center)
        self.press = self.circ.center, (event.xdata, event.ydata)
        self.dot_center=self.dot.get_center()
        
    def on_motion(self, event):
        """Move the circle if the mouse is over us."""
        if self.press is None or event.inaxes != self.circ.axes:
            return
        (x0, y0), (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.circ.set_center((x0+dx,y0+dy))
        self.cent.set_center((x0+dx,y0+dy))
        self.arc.set_center((x0+dx,y0+dy))
        self.line.set_data(x=x0+dx,y=y0+dy)
        x0_d,y0_d=self.dot_center
        self.dot.set_center((x0_d+dx,y0_d+dy))

        
        self.circ.figure.canvas.draw()

    def on_release(self, event):
        """Clear button press information."""
        self.press = None
        self.circ.figure.canvas.draw()

    def disconnect(self):
        """Disconnect all callbacks."""
        self.circ.figure.canvas.mpl_disconnect(self.cidpress)
        self.circ.figure.canvas.mpl_disconnect(self.cidrelease)
        self.circ.figure.canvas.mpl_disconnect(self.cidmotion)
        
        
        
        
class RotateCircle:
    def __init__(self, dot,circ,arc,line):
        self.dot = dot
        self.press = None
        self.arc = arc
        self.line = line
        self.circ = circ


    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.dot.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.dot.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.dot.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        """Check whether mouse is over us; if so, store some data."""
        if event.inaxes != self.dot.axes:
            return
        contains, attrd = self.dot.contains(event)
        if not contains:
            return
        print('event contains', self.dot.center)
        self.press = self.dot.center, (event.xdata, event.ydata)
        self.center=self.circ.get_center()

        
    def on_motion(self, event):
        """Move the circle if the mouse is over us."""
        if self.press is None or event.inaxes != self.dot.axes:
            return
        (x0, y0), (xpress, ypress) = self.press
        (xc,yc)=self.center
        v1 = [1, 0]
        v1 = v1/np.linalg.norm(v1)
        v2 = [event.xdata-xc,  event.ydata-yc]
        v2 = v2/np.linalg.norm(v2)
        if v2[1]<0:
            theta = 2*np.pi-np.arccos(np.dot(v1,v2))
        else: 
            theta = np.arccos(np.dot(v1,v2))
            
        self.dot.set_center((xc+1.5*v2[0],yc+1.5*v2[1]))
        self.arc.set_angle(-90+theta*180/np.pi)
        self.line.set_data(dx=1.5*v2[0],dy=1.5*v2[1])

        self.dot.figure.canvas.draw()

    def on_release(self, event):
        """Clear button press information."""
        self.press = None
        self.dot.figure.canvas.draw()

    def disconnect(self):
        """Disconnect all callbacks."""
        self.dot.figure.canvas.mpl_disconnect(self.cidpress)
        self.dot.figure.canvas.mpl_disconnect(self.cidrelease)
        self.dot.figure.canvas.mpl_disconnect(self.cidmotion)
        
        
class Index:
    
    def __init__(self, circles):
        self.circles = circles

    def alpha_type_pusher(self, event):
        self.circles[0].set_color("#00ffff")
        types[0]=-1

    def alpha_type_puller(self, event):
        self.circles[0].set_color("r")
        types[0]=1
        
    def alpha_type_neutral(self, event):
        self.circles[0].set_color("black")
        types[0]=0

    def beta_type_pusher(self, event):
        self.circles[1].set_color("#00ffff")
        types[1]=-1

    def beta_type_puller(self, event):
        self.circles[1].set_color("r")
        types[1]=1
        
    def beta_type_neutral(self, event):
        self.circles[1].set_color("black")
        types[1]=0
        
class Index_start:

 
    def start_button(self, event):
        global restart
        global pause
        global RESET
        if not thread.is_alive():          
            restart=False
            pause = False
            RESET = False
            thread.start()
        elif not pause:
            restart=True
            
        else:
            pause = not pause
            
    def pause_button(self, event):
        global pause
        pause = not pause
        
    def reset_button(self, event):
        global pause
        global RESET
        global types
        global objects, ax
        pause=True
        RESET=True
        types=[0,0]
        global thread
        
        ax.set_title(" Please locate swimmers, their orientation and specify their types.")
        for key in objects.keys()-['lines_assist','dots_assist']:
            for item in objects[key]:
                item.remove()
        add_objects([-1.5,0,np.pi/2,1.5,0,np.pi/2])
        drag_objects()
        bottom_buttons()
        
        thread = threading.Thread(target=start_trajectories, args=())