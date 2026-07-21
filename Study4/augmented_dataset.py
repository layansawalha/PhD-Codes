"""Hybrid stacking model"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

df = pd.read_csv("augmented_dataset_final.csv")
df.drop(columns=['contract_contract_sum', 'cost_increment'], inplace=True, errors='ignore')

# --- Leak-free split: first 2835 rows are the ORIGINAL data (test),
#     everything after is the AUGMENTED-only data (train). ---
N_ORIGINAL = 2835
original_df      = df.iloc[:N_ORIGINAL]
augmented_only_df = df.iloc[N_ORIGINAL:]

X_train = augmented_only_df.drop(columns=['cost_rebased'])
y_train = augmented_only_df['cost_rebased']
X_test  = original_df.drop(columns=['cost_rebased'])
y_test  = original_df['cost_rebased']

# Scaler fitted on TRAIN only, applied to test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

catboost = CatBoostRegressor(
    iterations=1000, depth=6, learning_rate=0.05,
    loss_function='RMSE', verbose=0, random_state=42,
)
catboost.fit(X_train, y_train)

xgb = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    objective='reg:squarederror', random_state=42,
)
xgb.fit(X_train, y_train)

estimators = [
    ('cat', catboost),
    ('xgb', xgb),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
        alpha=0.001, learning_rate='adaptive', max_iter=500, random_state=42,
    )),
]
stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    passthrough=True,
    n_jobs=-1,
)
stack_model.fit(X_train_scaled, y_train)
y_pred = stack_model.predict(X_test_scaled)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
print("--- Hybrid Stacking Ensemble Results (leak-free) ---")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

"""Neural Network Hybrid Model"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

df = pd.read_csv('augmented_dataset_final.csv')
df.dropna(inplace=True)

target = 'cost_rebased'

# --- Leak-free split by row position: original rows test, augmented rows train ---
N_ORIGINAL = 2835
original_df      = df.iloc[:N_ORIGINAL]
augmented_only_df = df.iloc[N_ORIGINAL:]

features = df.columns.drop(target)
X_train = augmented_only_df[features]
y_train = augmented_only_df[target]
X_test  = original_df[features]
y_test  = original_df[target]

# Scaler fitted on TRAIN only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Base learners fitted on TRAIN only, then used to produce meta-features
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                   subsample=0.8, random_state=42)
xgb.fit(X_train, y_train)
xgb_train_features = xgb.predict(X_train).reshape(-1, 1)
xgb_test_features  = xgb.predict(X_test).reshape(-1, 1)

cat = CatBoostRegressor(verbose=0, iterations=300, learning_rate=0.05,
                        depth=6, random_state=42)
cat.fit(X_train, y_train)
cat_train_features = cat.predict(X_train).reshape(-1, 1)
cat_test_features  = cat.predict(X_test).reshape(-1, 1)

mlp_input = Input(shape=(X_train_scaled.shape[1],), name='mlp_input')
x = Dense(128, activation='relu')(mlp_input)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
mlp_output = Dense(32, activation='relu')(x)

xgb_input = Input(shape=(1,), name='xgb_input')
cat_input = Input(shape=(1,), name='cat_input')

merged = Concatenate()([mlp_output, xgb_input, cat_input])
z = Dense(64, activation='relu')(merged)
z = Dropout(0.2)(z)
z = Dense(32, activation='relu')(z)
final_output = Dense(1, activation='linear')(z)

model = Model(inputs=[mlp_input, xgb_input, cat_input], outputs=final_output)
model.compile(optimizer='adam', loss='mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

early_stop = EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(
    [X_train_scaled, xgb_train_features, cat_train_features],
    y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
)

y_pred = model.predict([X_test_scaled, xgb_test_features, cat_test_features]).flatten()
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
print("\n--- Hybrid Deep Learning Model Results (leak-free) ---")
print(f"Test R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
