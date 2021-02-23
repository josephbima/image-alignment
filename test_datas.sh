declare -a arr=('beach' 'city' 'cp' 'desert' 'field' 'forest' 'hills' 'house' 'library' 'mountain' 'seaside' 'snow')

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
   # python main.py --ref ./3-dataset/"$i"_1.jpg --algn ./3-dataset/"$i"_2.jpg --matches 10000 --top 0.5 --out "$i"_12.jpg --debug True --ratio 4;
   python main.py --ref "$i"_12.jpg --algn ./3-dataset/"$i"_3.jpg --matches 10000 --top 0.5 --out "$i"_al.jpg --debug True --ratio 2

   # or do whatever with individual element of the array
done
