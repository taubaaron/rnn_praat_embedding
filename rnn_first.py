import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import praat_lpc_functions
# General Parameters
lpc_order = 64


# Generate data
sound = praat_lpc_functions.read_sound("/Users/aarontaub/Desktop/specific_nhss copy/f1_s3_song.wav")
lpc_in = praat_lpc_functions.lpc_extract(sound, lpc_order=lpc_order)
# praat_lpc_functions.plot_sound(sound)

sound = praat_lpc_functions.read_sound("/Users/aarontaub/Desktop/specific_nhss copy/f4_s3_song.wav")
lpc_out = praat_lpc_functions.lpc_extract(sound, lpc_order=lpc_order)


n = lpc_in.values.shape[1]  # number of frames

# Set window of past points for LSTM model
window = 10

# Split 80/20 into train/test data
"""last = int(n/5.0)
lpc_train = lpc_in.values[:, 0:last]
lpc_test = lpc_in.values[:, -last - window:]

lpc_train, lpc_test = np.array(lpc_train), np.array(lpc_test)"""
lpc_in_matrix, lpc_out_matrix = np.array(lpc_in.values), np.array(lpc_out.values)



lpc_in_matrix = lpc_in_matrix.reshape(lpc_in_matrix.shape[0], lpc_in_matrix.shape[1], 1)
lpc_out_matrix = lpc_out_matrix.reshape(lpc_out_matrix.shape[0], lpc_out_matrix.shape[1], 1)


# Initialize LSTM model
m = Sequential()
m.add(LSTM(units=50, return_sequences=True, input_shape=(lpc_in_matrix.shape[1],lpc_order)))
m.add(Dropout(0.2))
m.add(LSTM(units=50))
m.add(Dropout(0.2))
m.add(Dense(units=lpc_order))
m.compile(optimizer = 'adam', loss = 'mean_squared_error')


# Fit LSTM model
history = m.fit(lpc_in_matrix.T, lpc_out_matrix.T, epochs = 1, batch_size = 150,verbose=0)

plt.figure()
plt.ylabel('loss'); plt.xlabel('epoch')
plt.semilogy(history.history['loss'])
plt.show()


# TEST(?)
sound = praat_lpc_functions.read_sound("/Users/aarontaub/Desktop/specific_nhss copy/f1_s6_song.wav")
lpc_in_test = praat_lpc_functions.lpc_extract(sound, lpc_order=lpc_order)
lpc_in_matrix_test = np.array(lpc_in_test.values)
lpc_in_matrix_test = lpc_in_matrix_test.reshape(lpc_in_matrix_test.shape[0], lpc_in_matrix_test.shape[1], 1)

sound = praat_lpc_functions.read_sound("/Users/aarontaub/Desktop/specific_nhss copy/f4_s6_song.wav")
lpc_out_test = praat_lpc_functions.lpc_extract(sound, lpc_order=lpc_order)
lpc_out_matrix_test = np.array(lpc_out_test.values)
lpc_out_matrix_test = lpc_out_matrix_test.reshape(lpc_out_matrix_test.shape[0], lpc_out_matrix_test.shape[1], 1)


# Predict the next value (1 step ahead)
lpc_pred = m.predict(lpc_in_matrix_test.T)

# Plot prediction vs actual for test data
plt.figure()
plt.plot(lpc_pred,':',label='LSTM')
plt.plot(lpc_out_matrix_test,'--',label='Actual')
plt.legend()
plt.show()

# Plot prediction vs actual for test data
plt.figure()
plt.plot(lpc_pred.values,':',label='LSTM')
plt.plot(lpc_out_matrix_test.values,'--',label='Actual')
plt.legend()
plt.show()


# # Using predicted values to predict next step
# X_pred = lpc_test.copy()
# for i in range(window,len(X_pred)):
#     xin = X_pred[i-window:i].reshape((1, window, 1))
#     X_pred[i] = m.predict(xin)
#
# # Plot prediction vs actual for test data
# plt.figure()
# plt.plot(X_pred[window:],':',label='LSTM')
# plt.plot(next_X1,'--',label='Actual')
# plt.legend()
# plt.show()
#
# if __name__ == "__main__":
#     pass



"""
# Original
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

# Generate data
n = 500
t = np.linspace(0,20.0*np.pi,n)
X = np.sin(t)  # X is already between -1 and 1, scaling normally needed

# Set window of past points for LSTM model
window = 10

# Split 80/20 into train/test data
last = int(n/5.0)
Xtrain = X[:-last]
Xtest = X[-last-window:]

# Store window number of points as a sequence
xin = []
next_X = []
for i in range(window,len(Xtrain)):
    xin.append(Xtrain[i-window:i])
    next_X.append(Xtrain[i])

# Reshape data to format for LSTM
xin, next_X = np.array(xin), np.array(next_X)
xin = xin.reshape(xin.shape[0], xin.shape[1], 1)


# Initialize LSTM model
m = Sequential()
m.add(LSTM(units=50, return_sequences=True, input_shape=(xin.shape[1],1)))
m.add(Dropout(0.2))
m.add(LSTM(units=50))
m.add(Dropout(0.2))
m.add(Dense(units=1))
m.compile(optimizer = 'adam', loss = 'mean_squared_error')


# Fit LSTM model
history = m.fit(xin, next_X, epochs = 50, batch_size = 50,verbose=0)

plt.figure()
plt.ylabel('loss'); plt.xlabel('epoch')
plt.semilogy(history.history['loss'])


# Store "window" points as a sequence
xin = []
next_X1 = []
for i in range(window,len(Xtest)):
    xin.append(Xtest[i-window:i])
    next_X1.append(Xtest[i])

# Reshape data to format for LSTM
xin, next_X1 = np.array(xin), np.array(next_X1)
xin = xin.reshape((xin.shape[0], xin.shape[1], 1))

# Predict the next value (1 step ahead)
X_pred = m.predict(xin)

# Plot prediction vs actual for test data
plt.figure()
plt.plot(X_pred,':',label='LSTM')
plt.plot(next_X1,'--',label='Actual')
plt.legend()
plt.show()


# Using predicted values to predict next step
X_pred = Xtest.copy()
for i in range(window,len(X_pred)):
    xin = X_pred[i-window:i].reshape((1, window, 1))
    X_pred[i] = m.predict(xin)

# Plot prediction vs actual for test data
plt.figure()
plt.plot(X_pred[window:],':',label='LSTM')
plt.plot(next_X1,'--',label='Actual')
plt.legend()
plt.show()

if __name__ == "__main__":
    pass"""