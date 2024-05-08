import pyglet
import os

# parameters for the presentation of items.
ITEM_HEIGHT = 50
ITEM_WIDTH = 500

#gap between items. (In pixels.)
GAP = 10

DESELECT_COLOR = (100,100,100)
SELECT_COLOR = (255,170,0)

WINDOW_HEIGHT = 500
WINDOW_WIDTH = 750

# a single item in the list. can be selected and deselected.
class Item ():

    def __init__(self, xpos, ypos):
        self.box = pyglet.shapes.Rectangle(xpos, ypos, ITEM_WIDTH, ITEM_HEIGHT, color=DESELECT_COLOR)
        self.selected:bool = False

    def draw(self):
        self.box.draw()

    def select(self):
        self.selected = True
        self.box.color=SELECT_COLOR
        return self
        
    def deselect(self):
        self.selected = False
        self.box.color = DESELECT_COLOR
        return self

# a list of items.
class Stack ():

    def __init__(self, num_items, xpos, ypos):
        self.items:list[Item] = []
        self.cursor:int = 0 # position of the selected item in the stack.

        for i in range(num_items):
            self.items.append(Item(xpos, ypos + i * (ITEM_HEIGHT + GAP)))
        
        self.selected_item:Item = self.items[0].select() # selected item.

    def move_down (self):
        if self.cursor > 0:
            self.selected_item.deselect()
            self.cursor -= 1
            self.selected_item = self.items[self.cursor].select()
    
    def move_up (self):
        if self.cursor < len(self.items) - 1:
            self.selected_item.deselect()
            self.cursor += 1
            self.selected_item = self.items[self.cursor].select()

    def draw(self):
        for item in self.items:
            item.draw()

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

stack = Stack(8, 25, 25) # stack filled with arbitrary numbers

@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.UP:
        stack.move_up()
    if symbol == pyglet.window.key.DOWN:
        stack.move_down()
    if symbol == pyglet.window.key.Q:
        window.close()
        os._exit(0)


@window.event
def on_draw():
    window.clear()
    stack.draw()

pyglet.app.run()

