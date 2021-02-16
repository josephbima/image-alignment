declare -a arr=("book" "eg" "hill" "house" "kitchen" "park" "pier" "roof" "table")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
   python main.py --ref ./data/"$i"_1.jpg --algn ./data/"$i"_2.jpg --matches 10000 --top 0.3 --out "$i"_al.jpg --debug True
   # or do whatever with individual element of the array
done
