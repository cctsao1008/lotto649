// Exact four-level feature census over all 13,983,816 legal Lotto649 tickets.
// Frozen v0.16 graph code, implemented independently via six-node BFS.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
using namespace std;
struct Stat { uint64_t count=0, span_sum=0; array<uint64_t,49> indicator{}; };
static uint32_t graph_code(const array<int,6>& a) {
    int reach[6]={},card=0,diag=0;
    for(int i=0;i<6;i++)reach[i]=1<<i;
    for(int i=0;i<6;i++)for(int j=i+1;j<6;j++){
        int ri=(a[i]-1)/7,ci=(a[i]-1)%7,rj=(a[j]-1)/7,cj=(a[j]-1)%7;
        int dr=abs(ri-rj),dc=abs(ci-cj);
        if(dr<=1 && dc<=1){reach[i]|=1<<j;reach[j]|=1<<i;}
        if(dr+dc==1)++card;
        if(dr==1 && dc==1)++diag;
    }
    int unseen=63,components[6]={},count=0;
    while(unseen){
        int frontier=unseen & -unseen,found=0;
        while(frontier){
            int v=__builtin_ctz((unsigned)frontier);
            frontier&=frontier-1;
            frontier|=reach[v] & ~found;
            found|=1<<v;
        }
        if((found & ~unseen)!=0)throw runtime_error("BFS overlap");
        components[count++]=__builtin_popcount((unsigned)found);
        unseen &= ~found;
    }
    sort(components,components+6,greater<int>());
    uint32_t partition=0;
    for(int i=0;i<6;i++)partition|=(uint32_t)components[i]<<(3*(5-i));
    return (partition<<10)|((uint32_t)card<<5)|(uint32_t)diag;
}
int main(int argc,char**argv){
    if(argc==8 && string(argv[1])=="--classify"){
        array<int,6>a{};
        for(int i=0;i<6;i++){
            char*end=nullptr;long x=strtol(argv[i+2],&end,10);
            if(!end||*end||x<1||x>49||(i&&x<=a[i-1])){
                cerr<<"invalid sorted ticket\n";return 2;
            }
            a[i]=(int)x;
        }
        cout<<graph_code(a)<<"\n";return 0;
    }
    if(argc!=1){cerr<<"usage: census [--classify N1 ... N6]\n";return 2;}
    map<uint32_t,Stat>data;uint64_t total=0;array<int,6>a{};
    for(a[0]=1;a[0]<=44;a[0]++)
    for(a[1]=a[0]+1;a[1]<=45;a[1]++)
    for(a[2]=a[1]+1;a[2]<=46;a[2]++)
    for(a[3]=a[2]+1;a[3]<=47;a[3]++)
    for(a[4]=a[3]+1;a[4]<=48;a[4]++)
    for(a[5]=a[4]+1;a[5]<=49;a[5]++){
        auto&s=data[graph_code(a)];
        ++s.count;s.span_sum+=(uint64_t)(a[5]-a[0]);
        for(int n:a)++s.indicator[n-1];
        ++total;
    }
    if(total!=13983816||data.size()!=94)throw runtime_error("incomplete census");
    cout<<"#exact_covariance_v025.tsv.v1\t"<<total<<"\t"<<data.size()<<"\n";
    for(const auto &[code,s]:data){
        uint64_t num=0;for(auto v:s.indicator)num+=v;
        if(num!=6*s.count)throw runtime_error("invalid number conservation");
        cout<<code<<'\t'<<s.count<<'\t'<<s.span_sum;
        for(auto v:s.indicator)cout<<'\t'<<v;
        cout<<'\n';
    }
}
