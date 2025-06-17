#!/usr/bin/env python3
"""
临床transformer调试脚本
用于诊断和修复transformer模型在临床数据上的兼容性问题
"""

import torch
import numpy as np
from transformer import SpectraTransformer
from datasets import spectral_dataloader

def test_model_compatibility():
    """测试模型兼容性"""
    print("=" * 60)
    print("临床Transformer兼容性测试")
    print("=" * 60)
    
    # 1. 设备检测
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {DEVICE}")
    
    # 2. 创建8类临床模型
    print("\n创建8类临床模型...")
    clinical_model = SpectraTransformer(
        input_dim=1000,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        n_classes=8,  # 临床8类
        use_cls_token=True
    ).to(DEVICE)
    print(f"临床模型参数数量: {sum(p.numel() for p in clinical_model.parameters()):,}")
    
    # 3. 创建30类参考模型用于对比
    print("\n创建30类参考模型...")
    reference_model = SpectraTransformer(
        input_dim=1000,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        n_classes=30,  # 参考30类
        use_cls_token=True
    ).to(DEVICE)
    print(f"参考模型参数数量: {sum(p.numel() for p in reference_model.parameters()):,}")
    
    # 4. 检查权重兼容性
    print("\n检查权重兼容性...")
    clinical_dict = clinical_model.state_dict()
    reference_dict = reference_model.state_dict()
    
    compatible_layers = []
    incompatible_layers = []
    
    for key in clinical_dict.keys():
        if key in reference_dict:
            if clinical_dict[key].shape == reference_dict[key].shape:
                compatible_layers.append(key)
            else:
                incompatible_layers.append(f"{key}: {clinical_dict[key].shape} vs {reference_dict[key].shape}")
        else:
            incompatible_layers.append(f"{key}: 不存在于参考模型")
    
    print(f"兼容层数: {len(compatible_layers)}/{len(clinical_dict)}")
    print(f"不兼容层数: {len(incompatible_layers)}")
    
    if incompatible_layers:
        print("不兼容的层:")
        for layer in incompatible_layers:
            print(f"  - {layer}")
    
    # 5. 测试数据格式
    print("\n测试数据格式...")
    
    # 创建模拟临床数据
    batch_size = 5
    X_test = np.random.randn(10, 1000).astype(np.float32)
    y_test = np.random.randint(0, 8, 10)
    
    print(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")
    
    # 测试dataloader
    try:
        dl_test = spectral_dataloader(X_test, y_test, batch_size=batch_size, 
                                    shuffle=False, num_workers=0)
        print("✓ DataLoader创建成功")
        
        # 测试一个batch
        for inputs, targets in dl_test:
            print(f"DataLoader输出: inputs={inputs.shape}, targets={targets.shape}")
            print(f"数据类型: inputs={inputs.dtype}, targets={targets.dtype}")
            
            # 移动到设备
            if DEVICE == 'cuda':
                inputs = inputs.cuda()
                targets = targets.cuda()
            
            # 测试前向传播
            try:
                with torch.no_grad():
                    outputs = clinical_model(inputs)
                    print(f"模型输出形状: {outputs.shape}")
                    print("✓ 前向传播测试成功")
                    
                    # 测试损失计算
                    loss = torch.nn.CrossEntropyLoss()(outputs, targets.long())
                    print(f"损失值: {loss.item():.4f}")
                    print("✓ 损失计算测试成功")
                    
            except Exception as e:
                print(f"✗ 前向传播失败: {e}")
                return False
            break
            
    except Exception as e:
        print(f"✗ DataLoader测试失败: {e}")
        return False
    
    # 6. 测试权重加载兼容性
    print("\n测试权重加载...")
    try:
        # 尝试加载预训练权重
        try:
            ckpt = torch.load('./finetuned_transformer_model.ckpt', map_location=DEVICE)
            print(f"✓ 找到预训练模型，包含 {len(ckpt)} 层")
            
            # 测试兼容权重加载
            model_dict = clinical_model.state_dict()
            pretrained_dict = {k: v for k, v in ckpt.items() 
                             if k in model_dict and v.size() == model_dict[k].size()}
            
            print(f"可加载权重: {len(pretrained_dict)}/{len(ckpt)}")
            
            if pretrained_dict:
                model_dict.update(pretrained_dict)
                clinical_model.load_state_dict(model_dict)
                print("✓ 权重加载测试成功")
            else:
                print("⚠ 没有兼容的权重可加载")
                
        except FileNotFoundError:
            print("⚠ 预训练模型文件未找到，跳过权重加载测试")
            
    except Exception as e:
        print(f"✗ 权重加载测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("兼容性测试完成!")
    print("=" * 60)
    return True

def test_training_step():
    """测试训练步骤"""
    print("\n测试训练步骤...")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = SpectraTransformer(
        input_dim=1000, d_model=128, nhead=4, num_layers=4,
        dim_feedforward=512, dropout=0.1, n_classes=8, use_cls_token=True
    ).to(DEVICE)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建测试数据
    X_train = np.random.randn(20, 1000).astype(np.float32)
    y_train = np.random.randint(0, 8, 20)
    
    dl_train = spectral_dataloader(X_train, y_train, batch_size=5, 
                                 shuffle=True, num_workers=0)
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    try:
        for inputs, targets in dl_train:
            if DEVICE == 'cuda':
                inputs, targets = inputs.cuda(), targets.cuda()
            
            # 前向传播
            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, targets.long())
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data.long()).sum().item()
        
        acc = 100 * correct / total
        avg_loss = total_loss / len(dl_train)
        
        print(f"✓ 训练步骤测试成功")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  准确率: {acc:.2f}%")
        return True
        
    except Exception as e:
        print(f"✗ 训练步骤测试失败: {e}")
        return False

if __name__ == "__main__":
    success1 = test_model_compatibility()
    success2 = test_training_step()
    
    if success1 and success2:
        print("\n🎉 所有测试通过! 临床transformer应该可以正常工作。")
    else:
        print("\n❌ 某些测试失败，需要进一步调试。") 