import numpy as np
import torch
import matplotlib.pyplot as plt

def rse(input, target):
    return np.sqrt((input-target)**2)
def se(input, target):
    return (input-target)**2

# simulated data loaded
X_eval = np.load('data/simulated_data/rHeston/Xeval_grid=500_Hbeta_kappa1uniform_kappa2uniform_thetauniform_v0=0.01.npy')
Y_eval = np.load('data/simulated_data/rHeston/Yeval_grid=500_Hbeta_kappa1uniform_kappa2uniform_thetauniform_v0=0.01.npy')
X_torch = torch.from_numpy(X_eval).unsqueeze(1).float()

total_errors = {'Transformer': [], 'SigMA': [], 'Deepsignet': [], 'CNN': []}
total_merrors = {'Transformer': [], 'SigMA': [], 'Deepsignet': [], 'CNN': []}
RMSEs = {'Transformer': [], 'SigMA': [], 'Deepsignet': [], 'CNN': []}

for round_num in range(1, 11):
    # models loaded
    transformer = torch.load('data/results/multiple_rHeston/trained_models/round1/transformer.pth')
    sigma = torch.load('data/results/multiple_rHeston/trained_models/round1/sigma.pth')
    deepsignet = torch.load('data/results/multiple_rHeston/trained_models/round1/deepsignet.pth')
    cnn = torch.load('data/results/multiple_rHeston/trained_models/round1/cnn.pth')
    # models evaluation
    transformer.eval()
    sigma.eval()
    deepsignet.eval()
    cnn.eval()

    models = {'Transformer': transformer, 'SigMA': sigma,
              'Deepsignet': deepsignet, 'CNN': cnn}
    # prediction
    for model_name, model in models.items():
        predict = model(X_torch).detach().numpy()
        error = rse(predict, Y_eval)
        merror = np.mean(error, axis=1)
        rmse = np.mean(np.sqrt(np.mean(se(predict, Y_eval), axis=0)))

        total_errors[model_name].append(error)
        total_merrors[model_name].append(merror)
        RMSEs[model_name].append(rmse)
    print(f"已完成第{round_num}轮...")

for model_name in total_errors:
    total_errors[model_name] = np.array(total_errors[model_name])
    total_errors[model_name] = np.concatenate(total_errors[model_name], axis=0)
    total_merrors[model_name] = np.array(total_merrors[model_name])
    total_merrors[model_name] = np.concatenate(total_merrors[model_name], axis=0)
    RMSEs[model_name] = np.array(RMSEs[model_name])

# boxplot
box_data = [total_merrors['Transformer'], total_merrors['SigMA'],
            total_merrors['Deepsignet'], total_merrors['CNN']]
box_data_H = [total_errors['Transformer'][:,0], total_errors['SigMA'][:,0],
              total_errors['Deepsignet'][:,0], total_errors['CNN'][:,0]]
box_data_kappa1 = [total_errors['Transformer'][:,1], total_errors['SigMA'][:,1],
                total_errors['Deepsignet'][:,1], total_errors['CNN'][:,1]]
box_data_kappa2 = [total_errors['Transformer'][:,2], total_errors['SigMA'][:,2],
                total_errors['Deepsignet'][:,2], total_errors['CNN'][:,2]]
box_data_theta = [total_errors['Transformer'][:,3], total_errors['SigMA'][:,3],
                total_errors['Deepsignet'][:,3], total_errors['CNN'][:,3]]
labels = ['Transformer', 'SigMA', 'Deepsignet', 'CNN']

# boxplot of mean rse without fliers
plt.figure(figsize=(15,5))
plt.boxplot(box_data, #指定要绘制箱线图的数据
            whis = 1.5,
            vert = False,
            showmeans = True, #是否显示均值，默认值：False不显示
            meanline = True, #是否用线的形式表示均值，默认值False用点来表示
            showfliers = False, #是否显示异常值，默认值True显示；
            whiskerprops = {'linestyle':'--'}, #设置胡须的属性，如颜色、粗细、线的类型等
            labels = labels,
            flierprops = dict(markersize=5, marker='+'))
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=15)
plt.savefig('data/results/multiple_rHeston/errors_plots/rHeston_boxplot.eps')

# boxplot of mean rse with fliers
plt.figure(figsize=(15,5))
plt.boxplot(box_data, #指定要绘制箱线图的数据
            whis = 1.5,
            vert = False,
            showmeans = True, #是否显示均值，默认值：False不显示
            meanline = True, #是否用线的形式表示均值，默认值False用点来表示
            showfliers = True, #是否显示异常值，默认值True显示；
            whiskerprops = {'linestyle':'--'}, #设置胡须的属性，如颜色、粗细、线的类型等
            labels = labels,
            flierprops = dict(markersize=5, marker='+'))
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=15)
plt.savefig('data/results/multiple_rHeston/errors_plots/rHeston_boxplot_fliers.eps')

# output the RMSE and quantiles of average rse
for i, label in enumerate(labels):
    q1 = np.percentile(box_data[i], 25)
    q3 = np.percentile(box_data[i], 75)
    max_value = np.max(box_data[i])
    RMSE = np.mean(RMSEs[label])

    print(f"{label}:")
    print(f"Q1: {q1:.3f}")
    print(f"Q3: {q3:.3f}")
    print(f"Max: {max_value:.3f}")
    print(f"RMSE: {RMSE:.3f}")
    print()

# boxplot of rse for H without fliers
plt.figure(figsize=(15,5))
plt.boxplot(box_data_H, #指定要绘制箱线图的数据
            whis = 1.5,
            vert = False,
            showmeans = True, #是否显示均值，默认值：False不显示
            meanline = True, #是否用线的形式表示均值，默认值False用点来表示
            showfliers = False, #是否显示异常值，默认值True显示；
            whiskerprops = {'linestyle':'--'}, #设置胡须的属性，如颜色、粗细、线的类型等
            labels = labels,
            flierprops = dict(markersize=5, marker='+'))
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=15)
plt.savefig('data/results/multiple_rHeston/errors_plots/rHeston_boxplot_H.eps')

# boxplot of rse for H with fliers
plt.figure(figsize=(15,5))
plt.boxplot(box_data_H, #指定要绘制箱线图的数据
            whis = 1.5,
            vert = False,
            showmeans = True, #是否显示均值，默认值：False不显示
            meanline = True, #是否用线的形式表示均值，默认值False用点来表示
            showfliers = True, #是否显示异常值，默认值True显示；
            whiskerprops = {'linestyle':'--'}, #设置胡须的属性，如颜色、粗细、线的类型等
            labels = labels,
            flierprops = dict(markersize=5, marker='+'))
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=15)
plt.savefig('data/results/multiple_rHeston/errors_plots/rHeston_boxplot_H_fliers.eps')

# boxplot of rse for kappa1 without fliers
plt.figure(figsize=(15,5))
plt.boxplot(box_data_kappa1, #指定要绘制箱线图的数据
            whis = 1.5,
            vert = False,
            showmeans = True, #是否显示均值，默认值：False不显示
            meanline = True, #是否用线的形式表示均值，默认值False用点来表示
            showfliers = False, #是否显示异常值，默认值True显示；
            whiskerprops = {'linestyle':'--'}, #设置胡须的属性，如颜色、粗细、线的类型等
            labels = labels,
            flierprops = dict(markersize=5, marker='+'))
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=15)
plt.savefig('data/results/multiple_rHeston/errors_plots/rHeston_boxplot_kappa1.eps')

# boxplot of rse for kappa1 with fliers
plt.figure(figsize=(15,5))
plt.boxplot(box_data_kappa1, #指定要绘制箱线图的数据
            whis = 1.5,
            vert = False,
            showmeans = True, #是否显示均值，默认值：False不显示
            meanline = True, #是否用线的形式表示均值，默认值False用点来表示
            showfliers = True, #是否显示异常值，默认值True显示；
            whiskerprops = {'linestyle':'--'}, #设置胡须的属性，如颜色、粗细、线的类型等
            labels = labels,
            flierprops = dict(markersize=5, marker='+'))
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=15)
plt.savefig('data/results/multiple_rHeston/errors_plots/rHeston_boxplot_kappa1_fliers.eps')

# boxplot of rse for kappa2 without fliers
plt.figure(figsize=(15,5))
plt.boxplot(box_data_kappa2, #指定要绘制箱线图的数据
            whis = 1.5,
            vert = False,
            showmeans = True, #是否显示均值，默认值：False不显示
            meanline = True, #是否用线的形式表示均值，默认值False用点来表示
            showfliers = False, #是否显示异常值，默认值True显示；
            whiskerprops = {'linestyle':'--'}, #设置胡须的属性，如颜色、粗细、线的类型等
            labels = labels,
            flierprops = dict(markersize=5, marker='+'))
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=15)
plt.savefig('data/results/multiple_rHeston/errors_plots/rHeston_boxplot_kappa2.eps')

# boxplot of rse for kappa2 with fliers
plt.figure(figsize=(15,5))
plt.boxplot(box_data_kappa2, #指定要绘制箱线图的数据
            whis = 1.5,
            vert = False,
            showmeans = True, #是否显示均值，默认值：False不显示
            meanline = True, #是否用线的形式表示均值，默认值False用点来表示
            showfliers = True, #是否显示异常值，默认值True显示；
            whiskerprops = {'linestyle':'--'}, #设置胡须的属性，如颜色、粗细、线的类型等
            labels = labels,
            flierprops = dict(markersize=5, marker='+'))
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=15)
plt.savefig('data/results/multiple_rHeston/errors_plots/rHeston_boxplot_kappa2_fliers.eps')

# boxplot of rse for theta without fliers
plt.figure(figsize=(15,5))
plt.boxplot(box_data_theta, #指定要绘制箱线图的数据
            whis = 1.5,
            vert = False,
            showmeans = True, #是否显示均值，默认值：False不显示
            meanline = True, #是否用线的形式表示均值，默认值False用点来表示
            showfliers = False, #是否显示异常值，默认值True显示；
            whiskerprops = {'linestyle':'--'}, #设置胡须的属性，如颜色、粗细、线的类型等
            labels = labels,
            flierprops = dict(markersize=5, marker='+'))
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=15)
plt.savefig('data/results/multiple_rHeston/errors_plots/rHeston_boxplot_theta.eps')

# boxplot of rse for theta with fliers
plt.figure(figsize=(15,5))
plt.boxplot(box_data_theta, #指定要绘制箱线图的数据
            whis = 1.5,
            vert = False,
            showmeans = True, #是否显示均值，默认值：False不显示
            meanline = True, #是否用线的形式表示均值，默认值False用点来表示
            showfliers = True, #是否显示异常值，默认值True显示；
            whiskerprops = {'linestyle':'--'}, #设置胡须的属性，如颜色、粗细、线的类型等
            labels = labels,
            flierprops = dict(markersize=5, marker='+'))
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=15)
plt.savefig('data/results/multiple_rHeston/errors_plots/rHeston_boxplot_theta_fliers.eps')



