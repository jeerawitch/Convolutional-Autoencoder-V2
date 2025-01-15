# Convolutional_Autoencoder_Splite_Data VS Convolutional_Autoencoder_Best_Model

### ความแตกต่าง

1. การจัดการ Best Model

- Convolutional_Autoencoder_Splite_Data : ไม่มีการตรวจสอบหรือบันทึก best model

- Convolutional_Autoencoder_Best_Model : มีการตรวจสอบและบันทึกโมเดลที่มีค่าความแม่นยำ (Correct) สูงสุดลงในไฟล์ best_autoencoder.pth

```python
Best_acc = 0
Best_model = None
if Correct > Best_acc:
    Best_acc = Correct
    Best_model = model.state_dict()
    torch.save(Best_model, 'best_autoencoder.pth')
```

2.  การคำนวณ Accuracy

- Convolutional_Autoencoder_Splite_Data : ไม่มีการคำนวณค่าความแม่นยำ (Accuracy)

- Convolutional_Autoencoder_Best_Model : แปลงค่า Loss เป็นตัวบ่งชี้ความแม่นยำ (Accuracy) โดยคิดจากจำนวนพิกเซลของภาพ

```python
TotLossDiv = Loss / (64 * 64 * 3)
Correct = (1 - TotLossDiv) * 100
print(f'Train Epoch {epoch+1}/{num_epochs}, TotLossDiv: {TotLossDiv:.4f}, Accuracy: {Correct:.2f}%')

```

3. การบันทึกโมเดล

- Convolutional_Autoencoder_Splite_Data : บันทึกโมเดลเพียงครั้งเดียวหลังฝึกฝนจบ

```python
torch.save(model.state_dict(), 'conv_autoencoder.pth')
```

- Convolutional_Autoencoder_Best_Model : บันทึกเฉพาะโมเดลที่มีค่าความแม่นยำสูงสุด

```python
torch.save(Best_model, 'best_autoencoder.pth')
```

4. การโหลดโมเดลสำหรับการแสดงผล

- Convolutional_Autoencoder_Splite_Data : ใช้โมเดลที่ถูกฝึกครั้งสุดท้าย

```python
with torch.no_grad():
    for data, _ in test_loader_split:
        data = data.to(device)
        recon = model(data)
        break
```

- Convolutional_Autoencoder_Best_Model : โหลดและใช้โมเดลที่ดีที่สุดที่ถูกบันทึกไว้

```python
model.load_state_dict(torch.load('best_autoencoder.pth'))
model.eval()
with torch.no_grad():
    for data, _ in test_loader_split:
        data = data.to(device)
        recon = model(data)
        break
```

5. การแสดงผล Loss และ Accuracy

- Convolutional_Autoencoder_Splite_Data : แสดงเฉพาะค่า Loss

```python
print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
```

- Convolutional_Autoencoder_Best_Model : แสดงทั้ง Loss และ Accuracy เป็นเปอร์เซ็นต์

```python
print(f'Train Epoch {epoch+1}/{num_epochs}, TotLossDiv: {TotLossDiv:.4f}, Accuracy: {Correct:.2f}%')
```

6. การฝึกฝนและติดตามผล

- Convolutional_Autoencoder_Splite_Data : ไม่มีการติดตามผลระหว่างการฝึก

```python
for data in train_loader_split:
    img, _ = data
    img = img.to(device)
    optimizer.zero_grad()
    output = model(img)
    loss = criterion(output, img)
    loss.backward()
    optimizer.step()
```

- Convolutional_Autoencoder_Best_Model : ติดตามค่า Loss ระหว่างการฝึกเพื่อวิเคราะห์ในแต่ละ Epoch

```python
TrLoss = 0.0
for data in train_loader_split:
    img, _ = data
    img = img.to(device)
    optimizer.zero_grad()
    output = model(img)
    loss = criterion(output, img)
    loss.backward()
    optimizer.step()
    TrLoss += loss.item()
Loss = TrLoss / len(train_loader_split)
```

# Convolutional_Autoencoder_Best_Model VS Convolutional_Autoencoder_Best_Accuracy_And_Model

### ความแตกต่าง

1. การบันทึกโมเดล

- Convolutional_Autoencoder_Best_Model : บันทึกเฉพาะ state_dict ของโมเดลลงในไฟล์ best_autoencoder.pth โดยตรง

```python
torch.save(Best_model, 'best_autoencoder.pth')
```

- Convolutional_Autoencoder_Best_Accuracy_And_Model : ใช้ฟังก์ชัน save_best_model เพื่อบันทึกทั้ง state_dict ของโมเดลและค่าความแม่นยำ (accuracy) ลงในไฟล์ best_autoencoder.pth

```python
def save_best_model(model, accuracy, filepath):
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy
    }, filepath)

save_best_model(Best_model, Best_acc, filepath)
```

2. การโหลดโมเดล

- Convolutional_Autoencoder_Best_Model : โหลดเฉพาะ state_dict ของโมเดลจากไฟล์

```python
model.load_state_dict(torch.load('best_autoencoder.pth'))
```

- Convolutional_Autoencoder_Best_Accuracy_And_Model : โหลด state_dict และค่าความแม่นยำ (accuracy) จากไฟล์ และพิมพ์ค่าความแม่นยำที่โหลดมา

```python
checkpoint = torch.load(filepath)
model.load_state_dict(checkpoint['model_state_dict'])
Best_acc = checkpoint['accuracy']
print(f'Loaded Best Accuracy: {Best_acc:.2f}%')
```

3. การใช้ตัวแปรสำหรับจัดการโมเดล

- Convolutional_Autoencoder_Best_Model : เก็บเฉพาะ state_dict ของโมเดลที่ดีที่สุดในตัวแปร Best_model

```python
Best_model = model.state_dict()
```

- Convolutional_Autoencoder_Best_Accuracy_And_Model : เก็บทั้งตัวโมเดลที่ดีที่สุดไว้ในตัวแปร Best_model เพื่อให้สะดวกต่อการเรียกใช้งานฟังก์ชัน

```python
Best_model = model
```

4. การแยกฟังก์ชันสำหรับการบันทึกโมเดล

- Convolutional_Autoencoder_Best_Model : ไม่มีการแยกฟังก์ชันการบันทึกโมเดล

```python
torch.save(Best_model, 'best_autoencoder.pth')
```

- Convolutional_Autoencoder_Best_Accuracy_And_Model : แยกการบันทึกโมเดลออกเป็นฟังก์ชัน save_best_model เพื่อความเป็นระเบียบและใช้งานซ้ำได้

```python
def save_best_model(model, accuracy, filepath):
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy
    }, filepath)
```

5. การแสดงผลค่าความแม่นยำเมื่อโหลดโมเดล

- Convolutional_Autoencoder_Best_Model : ไม่มีการแสดงค่าความแม่นยำ (accuracy) หลังจากโหลดโมเดล

```python
model.load_state_dict(torch.load('best_autoencoder.pth'))
```

- Convolutional_Autoencoder_Best_Accuracy_And_Model : ดึงค่าความแม่นยำ (accuracy) จากไฟล์และแสดงผล

```python
checkpoint = torch.load(filepath)
Best_acc = checkpoint['accuracy']
print(f'Loaded Best Accuracy: {Best_acc:.2f}%')
```
