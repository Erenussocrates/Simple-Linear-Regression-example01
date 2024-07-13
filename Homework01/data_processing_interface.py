import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Not: Microsoft Excel'de ilk tüm tabloyu tarayıp Insert tab'ından Table'ı seçtikten sonra
# "My table has headers" alanını check ettim ki .csv dosyası doğru şekilde hazırlansın.

"""
.csv dosyasındakiler:

House Size (sqft)	Number of Bedrooms	Number of Bathrooms	Age of House (years)	Distance to City Center (miles)	    Price ($)
850	                2	                1	                20	                    15	                                120,000
900	                2	                1	                15	                    10	                                130,000
1000	            3	                2	                10	                    8	                                150,000
1200	            3	                2	                8	                    12	                                175,000
1500	            4	                3	                5	                    7	                                200,000
1800	            4	                3	                2	                    6	                                240,000
2000	            5	                4	                1	                    5	                                280,000
2200	            5	                4	                1	                    4	                                300,000
2400	            4	                3	                3	                    10	                                320,000
2600	            4	                3	                10	                    9	                                340,000
2800	            3	                2	                15	                    14	                                360,000
3000	            3	                2	                20	                    20	                                380,000
3200	            2	                2	                25                  	25	                                400,000
3400	            2	                2	                30	                    30	                                420,000
3600	            2	                3	                35	                    35	                                440,000

"""

my_data = pd.read_csv("Homework01.csv", delimiter=";")
# Burada delimiter=";" yaptık, çünkü genelde .csv dosyalarında değerler virgülle
# ayrılırken, benim export ettiğim dosyada noktalı virgülle ayrılmış, bu da
# okunması sırasında sorunlara yol açıyordu.

my_data['Log Price ($)'] = np.log(my_data['Price ($)'])
# Price'ın dağılımına baktıktan sonra, anlaşılıyor ki distribution'ımız normal bir
# distribution değil. Bu yüzden Chatgpt'nin önerisi doğrultusunda, 
# normal price sütununu bağımsız değişken olarak kullanmanın yanısıra,
# price'a log transformation uygulayıp onu da ayrıca bir bağımsız değişken
# olarak kullanmayı denemeye karar verdim.

while(True):
    user_choice=input("1) Null varmı bak\n2) Head'e bak\n3) Pairplot'a bak\n4) Distribution'lara bak\n5) Korelasyona bak\n")

    if(user_choice=="1"):
        print("My data table has any null field: ", end="")
        print(my_data.isnull().values.any())

        if(my_data.isnull().values.any()==True):
            print(my_data.isnull().sum().sum())

        print("Columns of the DataFrame: ", end="")
        print(my_data.columns)

        if my_data.empty:
            print("The DataFrame is empty.")
        else:
            print("The DataFrame is not empty.")
    elif(user_choice=="2"):
        print(my_data.head())
    elif(user_choice=="3"):
        pairplot = sns.pairplot(my_data)
        pairplot.figure.tight_layout(pad=2.0)
        pairplot.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
        plt.show()
    elif(user_choice=="4"):
        """sns.displot(my_data['Price ($)'])
        plt.show()
        sns.displot(my_data['Log Price ($)'])
        plt.show()
        sns.displot(my_data['Number of Bathrooms'])
        plt.show()"""
        # "sns.distplot" fonksiyonu deprecated olduğu için terminal "displot" veya "histplot"u
        # benzer kullanımlar için önerdi, kıyaslama için onlara baktım.

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot the distributions
        sns.histplot(my_data['Price ($)'], kde=True, ax=axes[0])
        axes[0].set_title('Price ($) Distribution')

        sns.histplot(my_data['Log Price ($)'], kde=True, ax=axes[1])
        axes[1].set_title('Log Price ($) Distribution')

        sns.histplot(my_data['Number of Bathrooms'], kde=True, ax=axes[2])
        axes[2].set_title('Number of Bathrooms Distribution')

        # Adjust the layout
        plt.tight_layout()

        # Show the plots
        plt.show()
    elif(user_choice=="5"):
        plt.figure(figsize=(18,10))
        heatmap=sns.heatmap(my_data.corr(),vmin=-1,vmax=1,annot=True)
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
        plt.show()
    else:
        break

### Çıkarımlar:
"""
1) Linear Regression'da bağımsız değişken olarak kullanılacak unsurun normal bir dağılımı olması gerekir.
Biz Price'ın ve Log Price'ın dağılımına bakarak, ikisinin de normal bir dağılımda olmadığını görmemizin yanısıra,
Linear Regression uygulamasıda kullanmış olmak adına bu iki independent variable'ı denemek için ayrı ayrı kullanabiliriz.
Bunun yanısıra, görünüşe göre "Number of Bathrooms" variable'ı normal bir distribution'a en yakındır.

2) Hem pairplot'lara hem de korelasyon ısı haritasına bakarak görebiliriz ki, "Price" ve "Log Price" variable'larının
öngörülmesi için en uygun olan variable "House Size"dır. Ki, "House Size"'ın, "Price" ile birebir korrelasyonu çıkmıştır,
ve umarım bu bazı inaccuracy'lere sebep olmaz.

Bunun yanında eğer independent variable olarak normal bir dağılıma sahip olan "Number of Bathrooms"u kullanacak isek,
bununla kullanılacak en uygun variable "Number of Bedrooms"dur.
"""