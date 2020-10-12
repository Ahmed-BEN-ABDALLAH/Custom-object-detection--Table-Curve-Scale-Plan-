import pyautogui
while True:
   pyautogui.moveTo(100, 150)
   pyautogui.moveRel(0, 500)  # move mouse 10 pixels down
   pyautogui.dragTo(100, 150)
   pyautogui.dragRel(0, 500)  # drag mouse 10 pixels down